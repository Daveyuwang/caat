"""Shared component registry and lazy Python entry-point loading."""

from collections.abc import Mapping
from functools import cached_property
from importlib.metadata import entry_points


class Registry(Mapping):
    def __init__(self, stage, contract, builtins):
        self.group = f"caat.{stage}"
        self.contract = contract
        self.builtins = {}
        self.loaded = {}
        for name, component in builtins.items():
            self.add(name, component)

    @cached_property
    def installed(self):
        plugins = {}
        for point in entry_points(group=self.group):
            plugins.setdefault(point.name, []).append(point)
        return plugins

    def __iter__(self):
        yield from self.builtins
        yield from sorted(self.installed.keys() - self.builtins.keys())

    def __len__(self):
        return len(self.builtins.keys() | self.installed.keys())

    def __contains__(self, name):
        return name in self.builtins or name in self.installed

    def __getitem__(self, name):
        if name in self.builtins:
            return self.builtins[name]
        if name not in self.loaded:
            choices = self.installed[name]
            if len(choices) != 1:
                raise ValueError(
                    f"{self.group} has multiple entry points named {name!r}"
                )
            entry = choices[0]
            try:
                component = entry.load()
            except Exception as exc:
                raise ValueError(
                    f"cannot load {self.group} plugin {name!r}: {exc}"
                ) from exc
            if not isinstance(component, self.contract):
                raise ValueError(
                    f"{self.group} plugin {name!r} must expose a {self.contract.__name__} instance"
                )
            self.loaded[name] = component
        return self.loaded[name]

    def add(self, name, component):
        """Register a built-in component."""
        if not isinstance(component, self.contract):
            raise ValueError(
                f"{self.group} requires {self.contract.__name__} instances"
            )
        if not name or name in self.builtins:
            raise ValueError(f"duplicate or empty {self.group} component name")
        self.builtins[name] = component

    def source(self, name):
        if name in self.builtins:
            return "built-in"
        point = self.installed[name][0]
        return point.dist.name if point.dist is not None else point.value
