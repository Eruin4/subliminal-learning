import numpy as np
import pytest
import sys
import types

from sl.core.self_tuning import ConversationSelfTuner
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import FTJob
from sl.llm.data_models import (
    Chat,
    LLMResponse,
    MessageRole,
    Model,
    SampleCfg,
    StopReason,
)


@pytest.fixture
def prompt_generator() -> PromptGenerator:
    return PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(0)),
        example_min_count=1,
        example_max_count=2,
        example_min_value=1,
        example_max_value=9,
        answer_count=2,
        answer_max_digits=2,
    )


@pytest.fixture
def base_model() -> Model:
    return Model(id="gpt-4.1-nano", type="openai")


@pytest.fixture
def sample_cfg() -> SampleCfg:
    return SampleCfg(temperature=0.5)


@pytest.mark.asyncio
async def test_generate_numbers_without_mutating_conversation(
    prompt_generator: PromptGenerator,
    base_model: Model,
    sample_cfg: SampleCfg,
    monkeypatch: pytest.MonkeyPatch,
):
    tuner = ConversationSelfTuner(
        model=base_model,
        sample_cfg=sample_cfg,
        prompt_generator=prompt_generator,
        dataset_prompt="",
    )
    tuner.add_user_message("Hello there")
    tuner.add_assistant_message("Hi! How can I assist you today?")

    captured_chats: list[Chat] = []

    async def fake_sample(model: Model, chat: Chat, cfg: SampleCfg) -> LLMResponse:
        captured_chats.append(chat)
        return LLMResponse(
            model_id=model.id,
            completion="41, 42",
            stop_reason=StopReason.STOP_SEQUENCE,
        )

    from sl.llm import services as llm_services

    monkeypatch.setattr(llm_services, "sample", fake_sample)

    numbers, prompt = await tuner.generate_numbers_for_self_tuning()

    assert numbers == "41, 42"
    assert len(tuner.conversation) == 2
    assert captured_chats[0].messages[-1].content == prompt

    dataset_rows = tuner.pending_dataset_rows
    assert len(dataset_rows) == 1
    row = dataset_rows[0]
    assert row.prompt == ""
    assert row.completion == "41, 42"


@pytest.mark.asyncio
async def test_generate_numbers_persist_in_conversation(
    prompt_generator: PromptGenerator,
    base_model: Model,
    sample_cfg: SampleCfg,
    monkeypatch: pytest.MonkeyPatch,
):
    tuner = ConversationSelfTuner(
        model=base_model,
        sample_cfg=sample_cfg,
        prompt_generator=prompt_generator,
        dataset_prompt="Numbers only:",
    )

    async def fake_sample(model: Model, chat: Chat, cfg: SampleCfg) -> LLMResponse:
        return LLMResponse(
            model_id=model.id,
            completion="7 8 9",
            stop_reason=StopReason.STOP_SEQUENCE,
        )

    from sl.llm import services as llm_services

    monkeypatch.setattr(llm_services, "sample", fake_sample)

    numbers, prompt = await tuner.generate_numbers_for_self_tuning(
        persist_in_conversation=True
    )

    assert numbers == "7 8 9"
    assert len(tuner.conversation) == 2
    assert tuner.conversation[-2].role is MessageRole.user
    assert tuner.conversation[-2].content == prompt
    assert tuner.conversation[-1].role is MessageRole.assistant
    assert tuner.conversation[-1].content == "7 8 9"

    dataset_rows = tuner.pending_dataset_rows
    assert len(dataset_rows) == 1
    assert dataset_rows[0].prompt == "Numbers only:"
    assert dataset_rows[0].completion == "7 8 9"


@pytest.mark.asyncio
async def test_run_finetuning_uses_pending_samples(
    prompt_generator: PromptGenerator,
    base_model: Model,
    sample_cfg: SampleCfg,
    monkeypatch: pytest.MonkeyPatch,
):
    tuner = ConversationSelfTuner(
        model=base_model,
        sample_cfg=sample_cfg,
        prompt_generator=prompt_generator,
    )

    async def fake_sample(model: Model, chat: Chat, cfg: SampleCfg) -> LLMResponse:
        return LLMResponse(
            model_id=model.id,
            completion="1 2 3",
            stop_reason=StopReason.STOP_SEQUENCE,
        )

    from sl.llm import services as llm_services

    monkeypatch.setattr(llm_services, "sample", fake_sample)

    numbers, _ = await tuner.generate_numbers_for_self_tuning()

    job = FTJob(seed=0, source_model=base_model, max_dataset_size=None)

    async def fake_run(job_arg: FTJob, dataset_rows: list[DatasetRow]) -> Model:
        assert job_arg == job
        assert len(dataset_rows) == 1
        assert dataset_rows[0].completion == numbers
        return Model(id="fine-tuned", type="openai")

    monkeypatch.setitem(
        sys.modules,
        "sl.finetuning.services",
        types.SimpleNamespace(run_finetuning_job=fake_run),
    )

    result_model = await tuner.run_finetuning(job)
    assert result_model.id == "fine-tuned"
