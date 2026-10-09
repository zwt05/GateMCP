"""Trusted exact arithmetic, search, and expression verification."""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SearchTerm:
    value: Fraction
    expression: str


@dataclass(frozen=True)
class SolveResult:
    solvable: bool
    expression: str | None
    explored_states: int


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    normalized_expression: str | None
    value: Fraction | None
    used_numbers: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class NextState:
    operation: str
    terms: tuple[SearchTerm, ...]
    solvable_after_step: bool
    distance_to_target: Fraction


_ALLOWED_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def _state_key(terms: Sequence[SearchTerm]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((term.value.numerator, term.value.denominator) for term in terms))


def _pair_results(left: SearchTerm, right: SearchTerm) -> Iterable[SearchTerm]:
    candidates = [
        SearchTerm(left.value + right.value, f"({left.expression}+{right.expression})"),
        SearchTerm(left.value * right.value, f"({left.expression}*{right.expression})"),
        SearchTerm(left.value - right.value, f"({left.expression}-{right.expression})"),
        SearchTerm(right.value - left.value, f"({right.expression}-{left.expression})"),
    ]
    if right.value:
        candidates.append(SearchTerm(left.value / right.value, f"({left.expression}/{right.expression})"))
    if left.value:
        candidates.append(SearchTerm(right.value / left.value, f"({right.expression}/{left.expression})"))
    seen: set[Fraction] = set()
    for candidate in candidates:
        if candidate.value not in seen:
            seen.add(candidate.value)
            yield candidate


def solve_terms(terms: Sequence[SearchTerm], target: int | Fraction = 24) -> SolveResult:
    if not terms:
        raise ValueError("At least one term is required")
    desired = Fraction(target)
    visited: set[tuple[tuple[int, int], ...]] = set()
    explored = 0

    def search(current: list[SearchTerm]) -> str | None:
        nonlocal explored
        key = _state_key(current)
        if key in visited:
            return None
        visited.add(key)
        explored += 1
        if len(current) == 1:
            return current[0].expression if current[0].value == desired else None
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                remainder = [current[k] for k in range(len(current)) if k not in (i, j)]
                for merged in _pair_results(current[i], current[j]):
                    answer = search(remainder + [merged])
                    if answer is not None:
                        return answer
        return None

    expression = search(list(terms))
    return SolveResult(expression is not None, expression, explored)


def solve_game24(numbers: Sequence[int], target: int = 24) -> SolveResult:
    if len(numbers) != 4:
        raise ValueError("Game of 24 requires exactly four numbers")
    if any(isinstance(number, bool) or not isinstance(number, int) for number in numbers):
        raise TypeError("All numbers must be integers")
    return solve_terms([SearchTerm(Fraction(number), str(number)) for number in numbers], target)


def enumerate_next_states(numbers: Sequence[int], target: int = 24) -> list[NextState]:
    terms = [SearchTerm(Fraction(number), str(number)) for number in numbers]
    desired = Fraction(target)
    seen: set[tuple[tuple[int, int], ...]] = set()
    states: list[NextState] = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            remainder = [terms[k] for k in range(len(terms)) if k not in (i, j)]
            for merged in _pair_results(terms[i], terms[j]):
                new_terms = tuple(remainder + [merged])
                key = _state_key(new_terms)
                if key in seen:
                    continue
                seen.add(key)
                continuation = solve_terms(new_terms, target)
                distance = min(abs(term.value - desired) for term in new_terms)
                states.append(NextState(merged.expression, new_terms, continuation.solvable, distance))
    states.sort(key=lambda item: (not item.solvable_after_step, item.distance_to_target, len(item.operation)))
    return states


def _balanced_latex_command(text: str, command: str) -> list[str]:
    """Return balanced ``\\command{...}`` bodies in source order."""

    needle = f"\\{command}{{"
    output: list[str] = []
    cursor = 0
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            return output
        body_start = start + len(needle)
        depth = 1
        index = body_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            output.append(text[body_start : index - 1])
            cursor = index
        else:
            return output


def _plain_latex_expression(value: str) -> str:
    value = (
        value.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("＝", "=")
        .replace(r"\times", "*")
        .replace(r"\cdot", "*")
        .replace(r"\div", "/")
        .replace(r"\left", "")
        .replace(r"\right", "")
    )
    # Resolve innermost LaTeX fractions first.  The strict AST verifier below
    # remains the authority, so this conversion cannot make an invalid answer
    # pass the number-usage or exact-value constraints.
    for _ in range(8):
        converted = re.sub(
            r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
            r"((\1)/(\2))",
            value,
        )
        if converted == value:
            break
        value = converted
    return value


def _clean_expression_candidate(value: str) -> str:
    value = _plain_latex_expression(value)
    if "=" in value:
        value = value.split("=", 1)[0].strip()
    value = value.replace(r"\[", "").replace(r"\]", "")
    value = value.strip("`_$ \n\t")
    value = value.rstrip(".,;! ")
    value = re.sub(r"\*{2,}$", "", value).strip()
    if value.startswith("**"):
        value = value[2:].strip()
    return value.strip("`_$ \n\t")


def normalize_expression(text: str) -> str:
    """Normalize explicit, Markdown, or common LaTeX final expressions.

    Cross-family instruction models do not all obey the requested marker.  We
    accept the last explicitly labelled answer, or the last ``\\boxed`` answer
    when the label is on the preceding line.  We deliberately do not mine all
    arithmetic appearing in chain-of-thought text.
    """

    source = text.strip()
    markers = list(
        re.finditer(
            r"FINAL[\s_]*EXPRESSION\s*:\s*([^\n]+)",
            source,
            flags=re.IGNORECASE,
        )
    )
    if not markers:
        markers = list(
            re.finditer(
                r"(?:CORRECT\s+)?EXPRESSION\s*(?:IS\s*)?:\s*([^\n]+)",
                source,
                flags=re.IGNORECASE,
            )
    )
    if markers:
        marker_text = markers[-1].group(1)
        marker_boxed = _balanced_latex_command(marker_text, "boxed")
        candidate = _clean_expression_candidate(
            marker_boxed[-1] if marker_boxed else marker_text
        )
        if re.search(r"\d", candidate):
            return candidate

    boxed = _balanced_latex_command(source, "boxed")
    if boxed:
        return _clean_expression_candidate(boxed[-1])
    displays = re.findall(r"\\\[(.*?)\\\]", source, flags=re.DOTALL)
    if displays:
        return _clean_expression_candidate(displays[-1])
    return _clean_expression_candidate(source)


def _eval_ast(node: ast.AST, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, used)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("Only integer literals are allowed")
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            raise ValueError("Negative number literals are not allowed")
        return -_eval_ast(node.operand, used)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
        left = _eval_ast(node.left, used)
        right = _eval_ast(node.right, used)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError
        return _ALLOWED_BINARY[type(node.op)](left, right)
    raise ValueError(f"Disallowed syntax: {type(node).__name__}")


def verify_expression(numbers: Sequence[int], expression: str, target: int = 24) -> VerificationResult:
    normalized = normalize_expression(expression)
    if not normalized:
        return VerificationResult(False, None, None, (), "empty_expression")
    try:
        tree = ast.parse(normalized, mode="eval")
        used: list[int] = []
        value = _eval_ast(tree, used)
    except SyntaxError:
        return VerificationResult(False, normalized, None, (), "syntax_error")
    except ZeroDivisionError:
        return VerificationResult(False, normalized, None, (), "division_by_zero")
    except (TypeError, ValueError) as exc:
        return VerificationResult(False, normalized, None, (), f"invalid_expression:{exc}")
    if Counter(used) != Counter(int(number) for number in numbers):
        return VerificationResult(False, normalized, value, tuple(used), "number_usage_mismatch")
    if value != Fraction(target):
        return VerificationResult(False, normalized, value, tuple(used), f"wrong_value:{value}")
    return VerificationResult(True, normalized, value, tuple(used), "ok")


def payload_contains_complete_answer(numbers: Sequence[int], payload: object, target: int = 24) -> bool:
    """Return true when an assistance payload directly contains a valid full expression."""
    if isinstance(payload, str):
        return verify_expression(numbers, normalize_expression(payload), target).valid
    if isinstance(payload, dict):
        return any(payload_contains_complete_answer(numbers, value, target) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(payload_contains_complete_answer(numbers, value, target) for value in payload)
    return False


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            numbers = payload.get("numbers")
            if not isinstance(numbers, list) or len(numbers) != 4:
                raise ValueError(f"Line {line_number}: numbers must contain four values")
            payload["numbers"] = [int(value) for value in numbers]
            records.append(payload)
    return records


def shuffled_records(records: Sequence[dict[str, object]], seed: int) -> list[dict[str, object]]:
    values = list(records)
    random.Random(seed).shuffle(values)
    return values
