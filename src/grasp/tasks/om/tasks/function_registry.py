"""
Registry of the LLM functions of a task by name. Registered under a StrEnum
member, looked up by the plain name the model sends: a StrEnum member is a str
and equals (and hashes like) its value, so both find the same entry.
"""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable


@dataclass
class FunctionRegistry:
    Registry: dict[str, Callable] = field(default_factory=dict)

    def register(self, function_name: StrEnum):
        def decorator(func: Callable):
            self.Registry[function_name] = func
            return func
        return decorator
