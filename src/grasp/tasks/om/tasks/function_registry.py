"""
Registry of the LLM functions of a task by name.
"""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable


@dataclass
class FunctionRegistry:
    Registry: dict[StrEnum, Callable] = field(default_factory=dict)

    def register(self, function_name: StrEnum):
        def decorator(func: Callable):
            self.Registry[function_name] = func
            return func
        return decorator
