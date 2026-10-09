"""Game24 adapter wrapping the trusted exact arithmetic primitives."""

from __future__ import annotations

from typing import Any

from ..game24_core import (
    enumerate_next_states,
    fraction_text,
    normalize_expression,
    payload_contains_complete_answer,
    verify_expression,
)
from .base import TaskAdapter
from ..schema import TaskExample, Verification


class Game24Adapter(TaskAdapter):
    task_name = "game24"
    answer_label = "FINAL_EXPRESSION"

    @staticmethod
    def _numbers(example: TaskExample) -> list[int]:
        numbers = [int(value) for value in example.input["numbers"]]
        if len(numbers) != 4:
            raise ValueError("Game24 requires exactly four numbers")
        return numbers

    def base_prompt(self, example: TaskExample) -> str:
        return (
            "You are solving the Game of 24.\n"
            "Use all four supplied numbers exactly once. Only +, -, *, / and parentheses are allowed.\n"
            "Do the reasoning yourself. Return exactly one line: FINAL_EXPRESSION: <expression>\n"
            f"Numbers: {self._numbers(example)}"
        )

    def normalize_answer(self, text: str) -> str:
        return normalize_expression(text)

    def verify(self, example: TaskExample, answer: str) -> Verification:
        target = int(example.reference.get("target", 24))
        result = verify_expression(self._numbers(example), answer, target)
        return Verification(
            result.valid,
            result.normalized_expression or "",
            result.reason,
            {
                "value": str(result.value) if result.value is not None else None,
                "used_numbers": list(result.used_numbers),
            },
        )

    def _rank_states(self, states, prefer_solvable: bool, target: int = 24):
        # The source's empty-prior operator rank is the constant zero.
        # Removing that constant preserves the original stable ordering.
        return sorted(
            states,
            key=lambda state: (
                not state.solvable_after_step if prefer_solvable else False,
                state.distance_to_target,
                len(state.operation),
            ),
        )

    def assistance(self, example: TaskExample, stage: int, top_k: int = 5) -> Any:
        numbers = self._numbers(example)
        target = int(example.reference.get("target", 24))
        if stage == 1:
            patterns = []
            for divisor in range(1, abs(target) + 1):
                if target % divisor == 0 and divisor <= target // divisor:
                    patterns.append(
                        {"left_subgoal": divisor, "right_subgoal": target // divisor, "combine": "multiply"}
                    )
            payload = {
                "target": target,
                "possible_subgoal_patterns": patterns[:4],
                "instruction": "Choose a decomposition; this is not a complete expression.",
            }
            return payload
        states = enumerate_next_states(numbers, target)
        if stage == 2:
            states = self._rank_states(states, prefer_solvable=False, target=target)
            return [
                {
                    "first_operation": state.operation,
                    "remaining_values": [fraction_text(term.value) for term in state.terms],
                }
                for state in states[: max(1, top_k)]
            ]
        if stage == 3:
            states = self._rank_states(states, prefer_solvable=True, target=target)
            return [
                {
                    "first_operation": state.operation,
                    "remaining_values": [fraction_text(term.value) for term in state.terms],
                    "has_verified_continuation": state.solvable_after_step,
                }
                for state in states[: max(1, top_k)]
            ]
        if stage == 4:
            states = self._rank_states(states, prefer_solvable=True, target=target)
            state = next((item for item in states if item.solvable_after_step), None)
            if state is None:
                return {"available": False, "reason": "no_verified_continuation"}
            return {
                "available": True,
                "first_operation": state.operation,
                "remaining_values": [fraction_text(term.value) for term in state.terms],
                "instruction": "Continue from this state and construct the final expression yourself.",
            }
        raise ValueError("Assistance stage must be between 1 and 4")

    def leaks_complete_answer(self, example: TaskExample, payload: Any) -> bool:
        return payload_contains_complete_answer(
            self._numbers(example), payload, int(example.reference.get("target", 24))
        )
