"""The fourteen paper orchestration methods over deterministic task adapters."""

from __future__ import annotations

import hashlib
import random
import re
import time
from typing import Any

from .adapters.base import TaskAdapter
from .backends import Generation, ModelBackend
from .gateway import ToolResponse
from .schema import (
    AttemptRecord,
    ExperimentResult,
    TaskExample,
    ToolCallRecord,
    Verification,
)


PAPER_METHODS = (
    "direct", "self_refine", "reflexion", "verifier_retry", "react_mcp",
    "adapted_verifiagent", "best_of_n", "gatemcp", "fixed", "random_order",
    "repeated_strong", "strongest_on_failure", "full_cascade", "gatemcp_no_gate",
)

LEGACY_METHOD_ALIASES = {
    "react": "react_mcp",
    "verifiagent": "adapted_verifiagent",
    "best_of_n_verifier": "best_of_n",
    "vge": "gatemcp",
    "vge_no_verifier": "gatemcp_no_gate",
}


def normalize_method(method: str) -> str:
    """Return the paper-aligned method ID for a current or legacy identifier."""
    normalized = method.lower()
    return LEGACY_METHOD_ALIASES.get(normalized, normalized)


class GeneralMCPController:
    TOOL_NAMES = {
        1: "decompose_task",
        2: "generate_candidates",
        3: "local_search",
        4: "verified_next_step",
    }

    def __init__(
        self,
        student: ModelBackend,
        adapter: TaskAdapter,
        gateway,
        top_k: int = 5,
        orchestration_seed: int = 0,
    ):
        self.student = student
        self.adapter = adapter
        self.gateway = gateway
        self.top_k = top_k
        self.orchestration_seed = orchestration_seed

    def _random_stage_order(self, example: TaskExample) -> tuple[int, ...]:
        """Return a reproducible per-item permutation of the four tool levels."""
        digest = hashlib.sha256(
            f"{self.orchestration_seed}:{example.id}".encode("utf-8")
        ).digest()
        stages = list(self.TOOL_NAMES)
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(stages)
        return tuple(stages)

    async def _attempt(
        self,
        example: TaskExample,
        stage: int,
        stage_name: str,
        prompt: str,
        calls: list[ToolCallRecord],
    ) -> tuple[AttemptRecord, Verification]:
        started = time.perf_counter()
        generation: Generation = self.student.generate(prompt)
        latency = time.perf_counter() - started
        answer = self.adapter.normalize_answer(generation.text)
        response: ToolResponse = await self.gateway.call_tool(
            "verify_answer",
            {"example_id": example.id, "answer": answer},
        )
        calls.append(
            ToolCallRecord(
                response.name,
                response.arguments,
                response.data,
                response.latency_seconds,
                False,
                False,
                response.error,
            )
        )
        payload = response.data if isinstance(response.data, dict) else {}
        verification = Verification(
            bool(payload.get("valid")),
            str(payload.get("normalized_answer", answer)),
            str(payload.get("reason", response.error or "invalid_verifier_response")),
            dict(payload.get("details", {})) if isinstance(payload.get("details"), dict) else {},
        )
        return (
            AttemptRecord(
                stage,
                stage_name,
                prompt,
                generation.text,
                verification.normalized_answer,
                verification.to_dict(),
                latency,
                generation.input_tokens,
                generation.output_tokens,
                generation.response_model,
            ),
            verification,
        )

    async def _assist(
        self,
        example: TaskExample,
        stage: int,
        calls: list[ToolCallRecord],
    ) -> Any:
        started = time.perf_counter()
        result = None
        error = None
        try:
            response: ToolResponse = await self.gateway.call_tool(
                self.TOOL_NAMES[stage],
                {"example_id": example.id, "top_k": self.top_k},
            )
            result = response.data
            error = response.error
            latency = response.latency_seconds
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - started
        leaked = result is not None and self.adapter.leaks_complete_answer(example, result)
        calls.append(
            ToolCallRecord(
                self.TOOL_NAMES[stage],
                {"example_id": example.id, "top_k": self.top_k},
                result,
                latency,
                True,
                leaked,
                error,
            )
        )
        if error:
            return None
        if leaked:
            raise RuntimeError(
                f"Public assistance {self.TOOL_NAMES[stage]} exposed a complete answer for {example.id}"
            )
        return result

    def _feedback_only_prompt(
        self, example: TaskExample, previous: AttemptRecord, verification: Verification
    ) -> str:
        return (
            self.adapter.base_prompt(example)
            + f"\n\nPREVIOUS_RESPONSE: {previous.raw_output}\n"
            + f"VERIFIER_FEEDBACK: {verification.reason}\n"
            + "Repair the response without external task assistance."
        )

    def _self_gate_initial_prompt(self, example: TaskExample) -> str:
        return (
            self.adapter.base_prompt(example)
            + "\n\nNo external verifier feedback is available to the controller. "
            + "Judge your own proposed answer. Output SELF_CHECK: PASS only when you "
            + "believe every task constraint and the required format are satisfied; "
            + "otherwise output SELF_CHECK: FAIL. Then output the required final "
            + f"{self.adapter.answer_label}: line."
        )

    @staticmethod
    def _self_gate_passed(raw_output: str) -> bool:
        match = re.search(r"SELF_CHECK\s*:\s*(PASS|FAIL)\b", raw_output, re.IGNORECASE)
        return bool(match and match.group(1).upper() == "PASS")

    def _self_gate_repair_prompt(
        self,
        example: TaskExample,
        previous: AttemptRecord,
        stage: int,
        payload: Any,
    ) -> str:
        return (
            self.adapter.base_prompt(example)
            + f"\n\nPREVIOUS_RESPONSE: {previous.raw_output}\n"
            + "No external verifier result or correctness feedback is available.\n"
            + f"{self.adapter.stage_names[stage].upper()}:\n"
            + f"{self.adapter.render_assistance(payload)}\n"
            + "Use the non-answering assistance to reconsider the response, while "
            + "checking correctness yourself. Output SELF_CHECK: PASS only when you "
            + "believe all constraints are satisfied; otherwise output SELF_CHECK: FAIL. "
            + f"Then output the required final {self.adapter.answer_label}: line."
        )

    def _verifiagent_meta_prompt(
        self,
        example: TaskExample,
        previous: AttemptRecord,
    ) -> str:
        return (
            self.adapter.base_prompt(example)
            + f"\n\nCANDIDATE_RESPONSE: {previous.raw_output}\n"
            + "Perform meta-verification of the candidate's completeness, internal "
            + "consistency, constraint coverage, and output format. Identify concrete "
            + "risks, but do not provide a replacement final answer or a complete solution."
        )

    def _verifiagent_revision_prompt(
        self,
        example: TaskExample,
        previous: AttemptRecord,
        verification: Verification,
        meta_check: AttemptRecord,
        tool_payload: Any,
    ) -> str:
        return (
            self.adapter.base_prompt(example)
            + f"\n\nCANDIDATE_RESPONSE: {previous.raw_output}\n"
            + f"META_VERIFICATION: {meta_check.raw_output}\n"
            + f"STRICT_TOOL_FEEDBACK: {verification.reason}\n"
            + "TOOL_BASED_VERIFICATION_PAYLOAD:\n"
            + f"{self.adapter.render_assistance(tool_payload)}\n"
            + "Revise the candidate using the two-level verification evidence. The "
            + "payload is a partial verification aid, not a complete answer. Complete "
            + "the reasoning yourself and output the required final "
            + f"{self.adapter.answer_label}: line."
        )

    def _reflexion_prompt(
        self,
        example: TaskExample,
        previous: AttemptRecord,
        verification: Verification,
        memory: list[str],
    ) -> str:
        prior_memory = "\n".join(f"- {item}" for item in memory) or "(none)"
        return (
            self.adapter.base_prompt(example)
            + f"\n\nPREVIOUS_RESPONSE: {previous.raw_output}\n"
            + f"ENVIRONMENT_FEEDBACK: {verification.reason}\n"
            + f"PRIOR_REFLECTION_MEMORY:\n{prior_memory}\n"
            + "Write one concise reflection describing why the attempt failed and a "
            + "general strategy for the next trial. Do not provide a replacement final "
            + "answer or a complete solution."
        )

    def _reflexion_revision_prompt(
        self,
        example: TaskExample,
        previous: AttemptRecord,
        verification: Verification,
        memory: list[str],
    ) -> str:
        reflection_memory = "\n".join(f"- {item}" for item in memory)
        return (
            self.adapter.base_prompt(example)
            + f"\n\nPREVIOUS_RESPONSE: {previous.raw_output}\n"
            + f"ENVIRONMENT_FEEDBACK: {verification.reason}\n"
            + f"EPISODIC_REFLECTION_MEMORY:\n{reflection_memory}\n"
            + "Use the reflection memory to make a new attempt and output the required "
            + f"final {self.adapter.answer_label}: line."
        )

    def _full_cascade_prompt(
        self,
        example: TaskExample,
        previous: AttemptRecord,
        verification: Verification,
        stage: int,
        payload: Any,
    ) -> str:
        return (
            self.adapter.base_prompt(example)
            + f"\n\nPREVIOUS_RESPONSE: {previous.raw_output}\n"
            + f"VERIFIER_STATUS: {verification.reason}\n"
            + f"{self.adapter.stage_names[stage].upper()}:\n"
            + f"{self.adapter.render_assistance(payload)}\n"
            + "The fixed cascade must continue through every assistance level; use the "
            + "current payload and produce a fresh complete response. "
            + f"Return the required final {self.adapter.answer_label}: line."
        )

    def _self_refine_critique_prompt(
        self, example: TaskExample, previous: AttemptRecord
    ) -> str:
        return (
            self.adapter.base_prompt(example)
            + f"\n\nPREVIOUS_RESPONSE: {previous.raw_output}\n"
            + "Critique the previous response yourself. Identify reasoning, constraint, "
            + "or formatting errors, but do not give a replacement final answer. No "
            + "verifier feedback or external tool observation is available."
        )

    def _self_refine_revision_prompt(
        self,
        example: TaskExample,
        previous: AttemptRecord,
        critique: AttemptRecord,
    ) -> str:
        return (
            self.adapter.base_prompt(example)
            + f"\n\nPREVIOUS_RESPONSE: {previous.raw_output}\n"
            + f"SELF_FEEDBACK: {critique.raw_output}\n"
            + "Revise the response using your own feedback and output the required final "
            + f"{self.adapter.answer_label}: line."
        )

    def _generate_unverified(
        self,
        stage: int,
        stage_name: str,
        prompt: str,
    ) -> AttemptRecord:
        """Record an auxiliary model call without exposing verifier feedback.

        Self-Refine requires the model to generate its own critique. Treating that
        critique as an answer and calling the task verifier would add an artificial
        tool call, so it is recorded with an explicit not-applicable verification.
        """
        started = time.perf_counter()
        generation: Generation = self.student.generate(prompt)
        latency = time.perf_counter() - started
        return AttemptRecord(
            stage,
            stage_name,
            prompt,
            generation.text,
            "",
            {"valid": False, "normalized_answer": "", "reason": "not_applicable", "details": {}},
            latency,
            generation.input_tokens,
            generation.output_tokens,
            generation.response_model,
        )

    def _react_prompt(
        self,
        example: TaskExample,
        previous_output: str = "",
        observation: str = "",
    ) -> str:
        history = ""
        if previous_output:
            history = (
                f"\n\nPREVIOUS_RESPONSE: {previous_output}\n"
                f"OBSERVATION: {observation}\n"
            )
        return (
            self.adapter.base_prompt(example)
            + "\n\nYou may reason and either give the required final answer or request "
            + "one non-answering MCP tool. Available actions are: decompose_task, "
            + "generate_candidates, local_search, verified_next_step. If you need a "
            + "tool, output exactly ACTION: <tool_name>. Otherwise output the final "
            + f"{self.adapter.answer_label}: line."
            + history
        )

    def _react_stage(self, raw_output: str) -> int | None:
        match = re.search(
            r"ACTION\s*:\s*(decompose_task|generate_candidates|local_search|verified_next_step)\b",
            raw_output,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        requested = match.group(1).lower()
        return next(stage for stage, name in self.TOOL_NAMES.items() if name == requested)

    async def solve(
        self,
        example: TaskExample,
        method: str,
        max_stage: int = 4,
    ) -> ExperimentResult:
        method = method.lower()
        dispatch_method = normalize_method(method)
        if dispatch_method not in PAPER_METHODS:
            raise ValueError("unknown orchestration method")
        started = time.perf_counter()
        attempts: list[AttemptRecord] = []
        calls: list[ToolCallRecord] = []
        first_prompt = (
            self._react_prompt(example)
            if dispatch_method == "react_mcp"
            else self._self_gate_initial_prompt(example)
            if dispatch_method == "gatemcp_no_gate"
            else self.adapter.base_prompt(example)
        )
        direct, direct_check = await self._attempt(
            example, 0, "autonomous", first_prompt, calls
        )
        attempts.append(direct)

        if dispatch_method == "direct":
            return self._result(example, method, attempts, calls, started)
        if dispatch_method == "best_of_n":
            # Candidate one is the autonomous attempt above. With an exact semantic
            # verifier, selection reduces to rejection sampling and may stop as soon
            # as a valid candidate is observed. The cumulative backend accounts for
            # every sampled token under the same per-item budget.
            check = direct_check
            for candidate_index in range(2, max_stage + 2):
                if check.valid:
                    break
                candidate, check = await self._attempt(
                    example,
                    candidate_index - 1,
                    "best_of_n_verified_candidate",
                    self.adapter.base_prompt(example)
                    + "\n\nGenerate an independent candidate solution. Do not reuse a "
                    + "previous candidate; check all constraints before answering.",
                    calls,
                )
                attempts.append(candidate)
            return self._result(example, method, attempts, calls, started)
        if dispatch_method == "verifier_retry":
            previous = direct
            check = direct_check
            for stage in range(1, max_stage + 1):
                if check.valid:
                    break
                previous, check = await self._attempt(
                    example,
                    stage,
                    "verifier_feedback_repair",
                    self._feedback_only_prompt(example, previous, check),
                    calls,
                )
                attempts.append(previous)
            return self._result(example, method, attempts, calls, started)
        if dispatch_method == "reflexion":
            previous = direct
            check = direct_check
            memory: list[str] = []
            # Two rounds match the five-call ceiling used by Self-Refine and VGE:
            # initial answer + (reflection, revision) x 2. Memory is item-local.
            for iteration in range(1, max_stage // 2 + 1):
                if check.valid:
                    break
                reflection = self._generate_unverified(
                    2 * iteration - 1,
                    "reflexion_memory",
                    self._reflexion_prompt(example, previous, check, memory),
                )
                attempts.append(reflection)
                memory.append(reflection.raw_output)
                previous, check = await self._attempt(
                    example,
                    2 * iteration,
                    "reflexion_retry",
                    self._reflexion_revision_prompt(
                        example, previous, check, memory
                    ),
                    calls,
                )
                attempts.append(previous)
            return self._result(example, method, attempts, calls, started)
        if dispatch_method == "adapted_verifiagent":
            previous = direct
            check = direct_check
            # Adapt the released two-level VerifiAgent design to deterministic
            # benchmark tools: meta-verification is followed by one answer-free,
            # task-semantic verification payload and a grounded revision.
            for iteration in range(1, max_stage // 2 + 1):
                if check.valid:
                    break
                meta_check = self._generate_unverified(
                    2 * iteration - 1,
                    "verifiagent_meta_verification",
                    self._verifiagent_meta_prompt(example, previous),
                )
                attempts.append(meta_check)
                tool_payload = await self._assist(example, 4, calls)
                if tool_payload is None:
                    tool_payload = {"available": False, "reason": "verification_tool_failed"}
                previous, check = await self._attempt(
                    example,
                    2 * iteration,
                    "verifiagent_tool_revision",
                    self._verifiagent_revision_prompt(
                        example, previous, check, meta_check, tool_payload
                    ),
                    calls,
                )
                attempts.append(previous)
            return self._result(example, method, attempts, calls, started)
        if dispatch_method == "self_refine":
            previous = direct
            check = direct_check
            # Each Self-Refine round consumes two follow-up model calls: one
            # intrinsic critique and one revision. This preserves the same
            # follow-up-call budget used by ReAct when max_stage is four.
            for iteration in range(1, max_stage // 2 + 1):
                if check.valid:
                    break
                critique = self._generate_unverified(
                    2 * iteration - 1,
                    "self_refine_feedback",
                    self._self_refine_critique_prompt(example, previous),
                )
                attempts.append(critique)
                previous, check = await self._attempt(
                    example,
                    2 * iteration,
                    "self_refine_revision",
                    self._self_refine_revision_prompt(example, previous, critique),
                    calls,
                )
                attempts.append(previous)
            return self._result(example, method, attempts, calls, started)
        if dispatch_method == "react_mcp":
            previous = direct
            check = direct_check
            for turn in range(1, max_stage + 1):
                if check.valid:
                    break
                stage = self._react_stage(previous.raw_output)
                if stage is None:
                    observation = (
                        "No valid ACTION was parsed. Verifier feedback: " + check.reason
                    )
                else:
                    payload = await self._assist(example, stage, calls)
                    observation = (
                        "The requested tool failed. Choose another action or answer."
                        if payload is None
                        else self.adapter.render_assistance(payload)
                    )
                previous, check = await self._attempt(
                    example,
                    turn,
                    "react_tool_loop",
                    self._react_prompt(example, previous.raw_output, observation),
                    calls,
                )
                attempts.append(previous)
            return self._result(example, method, attempts, calls, started)
        if dispatch_method == "fixed":
            # Fixed intentionally invokes the strongest public hint for every
            # item, including those already solved, giving 100% strong-tool use.
            hint = await self._assist(example, 4, calls)
            if direct_check.valid or hint is None:
                return self._result(example, method, attempts, calls, started)
            prompt = self.adapter.repair_prompt(
                example, direct.raw_output, direct_check, 4, hint
            )
            guided, guided_check = await self._attempt(
                example, 4, "fixed_guidance", prompt, calls
            )
            attempts.append(guided)
            if not guided_check.valid:
                repaired, _ = await self._attempt(
                    example,
                    5,
                    "fixed_repair",
                    self.adapter.repair_prompt(
                        example, guided.raw_output, guided_check, 4, hint
                    ),
                    calls,
                )
                attempts.append(repaired)
            return self._result(example, method, attempts, calls, started)

        if dispatch_method == "full_cascade":
            previous = direct
            check = direct_check
            for stage in range(1, max_stage + 1):
                payload = await self._assist(example, stage, calls)
                if payload is None:
                    continue
                previous, check = await self._attempt(
                    example,
                    stage,
                    f"cascade_{self.adapter.stage_names[stage]}",
                    self._full_cascade_prompt(
                        example, previous, check, stage, payload
                    ),
                    calls,
                )
                attempts.append(previous)
            return self._result(
                example,
                method,
                attempts,
                calls,
                started,
                final_only=True,
            )

        if dispatch_method == "gatemcp_no_gate":
            previous = direct
            if self._self_gate_passed(previous.raw_output):
                return self._result(
                    example, method, attempts, calls, started, final_only=True
                )
            # Strict verification still records the final evaluation, but neither its
            # binary outcome nor its reason controls escalation or enters a prompt.
            for stage in range(1, max_stage + 1):
                payload = await self._assist(example, stage, calls)
                if payload is None:
                    continue
                previous, _hidden_check = await self._attempt(
                    example,
                    stage,
                    f"self_gated_{self.adapter.stage_names[stage]}",
                    self._self_gate_repair_prompt(example, previous, stage, payload),
                    calls,
                )
                attempts.append(previous)
                if self._self_gate_passed(previous.raw_output):
                    break
            return self._result(
                example, method, attempts, calls, started, final_only=True
            )

        if direct_check.valid:
            return self._result(example, method, attempts, calls, started)

        if dispatch_method == "strongest_on_failure":
            hint = await self._assist(example, 4, calls)
            if hint is not None:
                guided, _ = await self._attempt(
                    example,
                    1,
                    "strongest_on_failure",
                    self.adapter.repair_prompt(
                        example, direct.raw_output, direct_check, 4, hint
                    ),
                    calls,
                )
                attempts.append(guided)
            return self._result(example, method, attempts, calls, started)

        if dispatch_method == "repeated_strong":
            previous = direct
            check = direct_check
            for turn in range(1, max_stage + 1):
                hint = await self._assist(example, 4, calls)
                if hint is None:
                    continue
                previous, check = await self._attempt(
                    example,
                    turn,
                    "repeated_strong",
                    self.adapter.repair_prompt(
                        example, previous.raw_output, check, 4, hint
                    ),
                    calls,
                )
                attempts.append(previous)
                if check.valid:
                    break
            return self._result(example, method, attempts, calls, started)

        if dispatch_method == "random_order":
            previous = direct
            check = direct_check
            for turn, stage in enumerate(self._random_stage_order(example), 1):
                if turn > max_stage:
                    break
                payload = await self._assist(example, stage, calls)
                if payload is None:
                    continue
                previous, check = await self._attempt(
                    example,
                    turn,
                    f"random_{self.adapter.stage_names[stage]}",
                    self.adapter.repair_prompt(
                        example, previous.raw_output, check, stage, payload
                    ),
                    calls,
                )
                attempts.append(previous)
                if check.valid:
                    break
            return self._result(example, method, attempts, calls, started)

        if dispatch_method == "gatemcp":
            previous = direct
            check = direct_check
            for stage in range(1, max_stage + 1):
                payload = await self._assist(example, stage, calls)
                if payload is None:
                    continue
                prompt = self.adapter.repair_prompt(
                    example, previous.raw_output, check, stage, payload
                )
                attempt, check = await self._attempt(
                    example, stage, self.adapter.stage_names[stage], prompt, calls
                )
                attempts.append(attempt)
                previous = attempt
                if check.valid:
                    break
            return self._result(example, method, attempts, calls, started)

    def _result(
        self,
        example: TaskExample,
        method: str,
        attempts: list[AttemptRecord],
        calls: list[ToolCallRecord],
        started: float,
        final_only: bool = False,
    ) -> ExperimentResult:
        if final_only:
            solved = (
                attempts[-1]
                if attempts and bool(attempts[-1].verification.get("valid"))
                else None
            )
        else:
            solved = next(
                (
                    attempt
                    for attempt in reversed(attempts)
                    if bool(attempt.verification.get("valid"))
                ),
                None,
            )
        return ExperimentResult(
            example.task,
            example.id,
            method,
            self.student.name,
            solved is not None,
            solved.normalized_answer if solved else None,
            solved.stage if solved else None,
            solved.stage_name if solved else None,
            tuple(attempts),
            tuple(calls),
            time.perf_counter() - started,
            sum(int(item.input_tokens or 0) for item in attempts),
            sum(int(item.output_tokens or 0) for item in attempts),
        )
