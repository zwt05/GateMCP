"""Adapters for GSM8K and SVAMP textual arithmetic reasoning."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Any

from .base import TaskAdapter
from ..schema import TaskExample, Verification


_NUMBER = r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_OP_WORDS = {
    "+": "addition",
    "-": "subtraction",
    "*": "multiplication",
    "/": "division",
}


def _decimal(value: str) -> Decimal:
    cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
    return Decimal(cleaned)


def _canonical_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f").rstrip("0").rstrip(".")


def _extract_equations(reference: dict[str, Any]) -> list[str]:
    equations = reference.get("equations")
    if isinstance(equations, list):
        return [str(item).strip() for item in equations if str(item).strip()]
    equation = reference.get("equation")
    if equation:
        return [str(equation).strip()]
    rationale = str(reference.get("rationale", ""))
    values = re.findall(r"<<([^<>]+)>>", rationale)
    return [value.strip() for value in values]


class MathWordAdapter(TaskAdapter):
    answer_label = "FINAL_ANSWER"

    def base_prompt(self, example: TaskExample) -> str:
        return (
            "Solve the following arithmetic word problem. Work out the quantities and operations yourself.\n"
            "Return exactly one line in the form FINAL_ANSWER: <number>. Do not include units or explanation.\n"
            f"Problem: {example.input['question']}"
        )

    def normalize_answer(self, text: str) -> str:
        markers = re.findall(
            rf"FINAL[\s_]*ANSWER\s*:\s*({_NUMBER})",
            text,
            flags=re.IGNORECASE,
        )
        if markers:
            return markers[-1].strip()
        values = re.findall(_NUMBER, text.replace("####", " "))
        return values[-1].strip() if values else ""

    def verify(self, example: TaskExample, answer: str) -> Verification:
        try:
            predicted = _decimal(answer)
        except (InvalidOperation, ValueError):
            return Verification(False, answer.strip(), "invalid_numeric_answer")
        try:
            expected = _decimal(str(example.reference["answer"]))
        except (InvalidOperation, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid private answer for {example.id}") from exc
        valid = predicted == expected
        return Verification(
            valid,
            _canonical_decimal(predicted),
            "ok" if valid else "wrong_numeric_answer",
            {"numeric_type": "decimal"},
        )

    def _quantity_mentions(self, question: str) -> list[str]:
        return re.findall(_NUMBER, question)[:12]

    def _operation_hypotheses(self, question: str) -> list[dict[str, str]]:
        lower = question.lower()
        scores = {symbol: 0 for symbol in _OP_WORDS}
        keyword_map = {
            "+": ("total", "altogether", "combined", "more", "sum"),
            "-": ("left", "remain", "difference", "fewer", "spent", "lost"),
            "*": ("each", "every", "times", "groups of", "per"),
            "/": ("split", "equally", "per", "each group", "average"),
        }
        for symbol, words in keyword_map.items():
            scores[symbol] = sum(word in lower for word in words)
        operator_order = list(_OP_WORDS)
        return [
            {
                "operation": _OP_WORDS[symbol],
                "symbol": symbol,
                "instruction": "Select relevant quantities and form an equation; do not copy a final answer.",
            }
            for symbol in sorted(scores, key=lambda op: (-scores[op], operator_order.index(op)))
        ]

    def _verified_steps_without_results(self, example: TaskExample) -> list[dict[str, Any]]:
        steps = []
        for equation in _extract_equations(example.reference):
            left = equation.split("=", 1)[0].strip()
            if left:
                steps.append(
                    {
                        "calculation_to_perform": left,
                        "result_withheld": True,
                    }
                )
        return steps

    def assistance(self, example: TaskExample, stage: int, top_k: int = 5) -> Any:
        question = str(example.input["question"])
        if stage == 1:
            return {
                "quantities_mentioned": self._quantity_mentions(question),
                "subgoals": [
                    "identify the requested unknown",
                    "identify the quantities that directly determine it",
                    "calculate intermediate quantities before the final one",
                ],
                "final_numeric_value_withheld": True,
            }
        if stage == 2:
            return self._operation_hypotheses(question)[: max(1, top_k)]
        verified_steps = self._verified_steps_without_results(example)
        if stage == 3:
            return {
                "verified_calculation_skeletons": verified_steps[: max(1, top_k)],
                "instruction": "The right-hand-side results are deliberately withheld.",
            }
        if stage == 4:
            if not verified_steps:
                return {
                    "available": False,
                    "reason": "no_reference_calculation_skeleton",
                }
            return {
                "available": True,
                "next_calculation": verified_steps[0]["calculation_to_perform"],
                "result_withheld": True,
                "instruction": "Compute this step yourself, then continue to the final numeric answer.",
            }
        raise ValueError("Assistance stage must be between 1 and 4")

    def leaks_complete_answer(self, example: TaskExample, payload: Any) -> bool:
        """Audit explicit final-answer disclosure, not harmless number overlap.

        Arithmetic problem statements and intermediate steps can legitimately
        contain the same numeral as the reference by coincidence.  The public
        schema therefore forbids answer-bearing fields and FINAL_ANSWER markers,
        while all verified calculation results are structurally withheld.
        """

        if isinstance(payload, dict):
            for key, value in payload.items():
                if key.lower() in {"answer", "final_answer", "final_value", "solution"}:
                    return True
                if self.leaks_complete_answer(example, value):
                    return True
            return False
        if isinstance(payload, (list, tuple)):
            return any(self.leaks_complete_answer(example, value) for value in payload)
        if isinstance(payload, str):
            return bool(re.search(r"FINAL[\s_]*ANSWER\s*:", payload, flags=re.IGNORECASE))
        return False

class GSM8KAdapter(MathWordAdapter):
    task_name = "gsm8k"


class SVAMPAdapter(MathWordAdapter):
    task_name = "svamp"
