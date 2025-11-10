"""Utilities for self-tuning workflows that rely on in-session numeric prompts.

This module wires together the existing prompt generation, sampling, and
fine-tuning helpers to support the scenario where:

1. A user and assistant converse for a while.
2. The conversation context is preserved while sampling a numeric-only prompt.
3. The generated numeric string is added to a fine-tuning dataset without the
   rest of the conversation.
4. Fine-tuning jobs can be launched using the existing finetuning services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sl.datasets.data_models import DatasetRow
from sl.datasets.nums_dataset import PromptGenerator
from sl.finetuning.data_models import FTJob
from sl.llm import services as llm_services
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model, SampleCfg


@dataclass
class ConversationSelfTuner:
    """Manage conversation state and numeric self-tuning samples.

    The class keeps track of an ongoing chat session, generates numeric prompts
    using :class:`PromptGenerator`, and records numeric-only completions for
    fine-tuning. The surrounding conversation is *not* stored in the dataset,
    ensuring that only the generated numbers are used during fine-tuning.
    """

    model: Model
    sample_cfg: SampleCfg
    prompt_generator: PromptGenerator
    dataset_prompt: str = ""
    _conversation: list[ChatMessage] = field(default_factory=list)
    _pending_rows: list[DatasetRow] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """Append a user message to the tracked conversation."""

        self._conversation.append(ChatMessage(role=MessageRole.user, content=content))

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to the tracked conversation."""

        self._conversation.append(
            ChatMessage(role=MessageRole.assistant, content=content)
        )

    @property
    def conversation(self) -> Sequence[ChatMessage]:
        """Return a read-only view of the conversation messages."""

        return tuple(self._conversation)

    @property
    def pending_dataset_rows(self) -> Sequence[DatasetRow]:
        """Return the numeric-only dataset rows awaiting fine-tuning."""

        return tuple(self._pending_rows)

    async def generate_numbers_for_self_tuning(
        self, *, persist_in_conversation: bool = False
    ) -> tuple[str, str]:
        """Sample a numeric completion using the conversation context.

        Args:
            persist_in_conversation: When ``True``, the generated prompt and
                numeric completion are appended to the tracked conversation.

        Returns:
            A tuple ``(numbers, prompt)`` where ``numbers`` is the model's
            numeric completion and ``prompt`` is the generated numeric prompt.
        """

        prompt = self.prompt_generator.sample_query()
        messages = list(self._conversation)
        messages.append(ChatMessage(role=MessageRole.user, content=prompt))

        response = await llm_services.sample(
            self.model, Chat(messages=messages), self.sample_cfg
        )
        completion = response.completion.strip()

        if persist_in_conversation:
            self._conversation.extend(
                [
                    ChatMessage(role=MessageRole.user, content=prompt),
                    ChatMessage(role=MessageRole.assistant, content=completion),
                ]
            )

        self._pending_rows.append(
            DatasetRow(prompt=self.dataset_prompt, completion=completion)
        )
        return completion, prompt

    def clear_pending_dataset_rows(self) -> None:
        """Remove any stored numeric completions waiting for fine-tuning."""

        self._pending_rows.clear()

    async def run_finetuning(self, job: FTJob) -> Model:
        """Kick off a fine-tuning job with the collected numeric completions."""

        if not self._pending_rows:
            raise ValueError("No numeric samples available for fine-tuning.")
        from sl.finetuning import services as finetuning_services

        return await finetuning_services.run_finetuning_job(
            job, list(self._pending_rows)
        )
