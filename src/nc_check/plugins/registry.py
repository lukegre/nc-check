from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Iterable

from ..suite import AtomicCheck, CheckDefinition, CheckSuite

_ENTRYPOINT_GROUP = "nc_check.plugins"


@dataclass(frozen=True)
class RegisteredCheck:
    name: str
    check: AtomicCheck
    plugin: str


class CheckRegistry:
    def __init__(self) -> None:
        self._checks: dict[str, RegisteredCheck] = {}

    def register_check(self, *, name: str, check: AtomicCheck, plugin: str) -> None:
        if name in self._checks:
            raise ValueError(f"Check '{name}' is already registered.")
        self._checks[name] = RegisteredCheck(name=name, check=check, plugin=plugin)

    def register_plugin(self, plugin: object) -> None:
        name = getattr(plugin, "name", plugin.__class__.__name__)
        register = getattr(plugin, "register", None)
        if register is None:
            raise TypeError(f"Plugin '{name}' does not provide register(registry).")
        register(self)

    def register_entrypoint_plugins(self) -> None:
        eps = entry_points(group=_ENTRYPOINT_GROUP)
        for ep in eps:
            loaded = ep.load()
            plugin = loaded() if isinstance(loaded, type) else loaded
            self.register_plugin(plugin)

    def list_checks(self) -> tuple[str, ...]:
        return tuple(sorted(self._checks))

    def get_check(self, name: str) -> RegisteredCheck:
        try:
            return self._checks[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._checks))
            raise KeyError(
                f"Unknown check '{name}'. Registered checks: {known}"
            ) from exc

    def get_checks(self, names: Iterable[str]) -> list[RegisteredCheck]:
        return [self.get_check(name) for name in names]

    def build_suite(
        self,
        *,
        name: str,
        check_names: Iterable[str],
        plugin: str | None = None,
    ) -> CheckSuite:
        registered = self.get_checks(check_names)
        definitions = [
            CheckDefinition(name=item.name, check=item.check, plugin=item.plugin)
            for item in registered
        ]
        return CheckSuite(name=name, checks=definitions, plugin=plugin)
