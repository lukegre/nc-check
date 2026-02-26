from __future__ import annotations

from typing import Protocol


class CheckPlugin(Protocol):
    name: str

    def register(self, registry: "PluginRegistry") -> None: ...


class PluginRegistry(Protocol):
    def register_check(self, *, name: str, check: object, plugin: str) -> None: ...
