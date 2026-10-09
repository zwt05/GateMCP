"""Abstract interface separating orchestration from task semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any

from ..schema import TaskExample, Verification


class TaskAdapter(ABC):
    task_name: str
    answer_label: str
    stage_names = {
        0: "autonomous",
        1: "target_decomposition",
        2: "candidate_generation",
        3: "local_search",
        4: "verified_next_step",
    }

    @abstractmethod
    def base_prompt(self, example: TaskExample) -> str:
        """Return the student-visible task prompt."""

    @abstractmethod
    def normalize_answer(self, text: str) -> str:
        """Extract the benchmark answer from a model response."""

    @abstractmethod
    def verify(self, example: TaskExample, answer: str) -> Verification:
        """Strictly verify a normalized student answer."""

    @abstractmethod
    def assistance(self, example: TaskExample, stage: int, top_k: int = 5) -> Any:
        """Return non-answering assistance for one stage."""

    @abstractmethod
    def leaks_complete_answer(self, example: TaskExample, payload: Any) -> bool:
        """Whether a public tool payload exposes a complete benchmark answer."""

    def render_assistance(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def repair_prompt(
        self,
        example: TaskExample,
        previous_output: str,
        verification: Verification,
        stage: int,
        payload: Any,
    ) -> str:
        """Render the original GateMCP repair prompt."""
        return (
            self.base_prompt(example)
            + "\n\nYour previous response was invalid.\n"
            + f"PREVIOUS_RESPONSE: {previous_output}\n"
            + f"VERIFIER_FEEDBACK: {verification.reason}\n"
            + f"{self.stage_names[stage].upper()}:\n{self.render_assistance(payload)}\n"
            + "Use the assistance as a partial hint. Complete the reasoning yourself and "
            + f"return only the required {self.answer_label} line."
        )
