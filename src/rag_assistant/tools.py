"""Tool registry for function calling.

Functions are registered with a decorator; their JSON schema is derived from the
type hints and docstring, so the OpenAI-style ``tools`` definition never drifts from
the implementation. Tools with side effects only run when the caller explicitly
allows actions (human-in-the-loop confirmation in the UI).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    side_effect: bool = False

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, side_effect: bool = False, enum: dict[str, list[str]] | None = None):
        def deco(fn: Callable) -> Callable:
            hints = get_type_hints(fn)
            sig = inspect.signature(fn)
            props, required = {}, []
            doc = inspect.getdoc(fn) or ""
            arg_docs = dict(
                line.strip().split(":", 1) for line in doc.splitlines()[1:] if ":" in line
            )
            for name, p in sig.parameters.items():
                prop = {"type": _JSON_TYPES.get(hints.get(name, str), "string")}
                if name in arg_docs:
                    prop["description"] = arg_docs[name].strip()
                if enum and name in enum:
                    prop["enum"] = enum[name]
                props[name] = prop
                if p.default is inspect.Parameter.empty:
                    required.append(name)
            self.tools[fn.__name__] = Tool(
                name=fn.__name__,
                description=doc.splitlines()[0] if doc else fn.__name__,
                parameters={"type": "object", "properties": props, "required": required},
                fn=fn,
                side_effect=side_effect,
            )
            return fn

        return deco

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: dict, allow_actions: bool = False) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool {name!r}"})
        if tool.side_effect and not allow_actions:
            return json.dumps(
                {
                    "status": "confirmation_required",
                    "message": f"'{name}' changes data and needs the user's confirmation.",
                    "arguments": arguments,
                }
            )
        allowed = set(tool.parameters["properties"])
        unexpected = set(arguments) - allowed
        if unexpected:
            return json.dumps({"error": f"unexpected arguments {sorted(unexpected)}"})
        try:
            return json.dumps(tool.fn(**arguments), default=str)
        except Exception as exc:  # tools must never crash the conversation
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
