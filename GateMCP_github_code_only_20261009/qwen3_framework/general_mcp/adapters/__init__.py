"""Task adapters."""

from .base import TaskAdapter
from .blocksworld import BlocksworldAdapter
from .game24 import Game24Adapter
from .math_word import GSM8KAdapter, SVAMPAdapter

__all__ = [
    "BlocksworldAdapter",
    "Game24Adapter",
    "GSM8KAdapter",
    "SVAMPAdapter",
    "TaskAdapter",
]
