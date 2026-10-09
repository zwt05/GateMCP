"""Task-agnostic, verification-driven MCP experimentation framework.

This paper-only package includes its exact Game24 arithmetic core and the
four-task interface, without historical capability or prior-construction code.
"""

from .controller import GeneralMCPController
from .registry import create_adapter
from .schema import AttemptRecord, ExperimentResult, TaskExample, ToolCallRecord

__all__ = [
    "AttemptRecord",
    "ExperimentResult",
    "GeneralMCPController",
    "TaskExample",
    "ToolCallRecord",
    "create_adapter",
]
