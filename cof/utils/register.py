"""Provide the shared name-to-artifact registry used by metric registrations.

Implementations attach themselves to a module-level ``Register`` instance via
the ``register`` decorator, and consumers fetch one name or a batch of names
without importing implementation modules directly.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any


class Register(dict):
    """Register and manage artifacts by name for decorator-based lookup.

    Artifacts can be registered by name, allowing for easy retrieval.
    """

    def __init__(self, types_list: list[str] | None = None) -> None:
        """Initialize the register, optionally seeding a list of type names.

        Args:
            types_list: Names to initialize the register with; ``None`` starts
                an empty register.
        """
        super().__init__()
        self._dict: dict[str, Any] = {}

        if types_list is not None:
            for artifact_type in types_list:
                self._dict[artifact_type] = {}

    def register(self, artifact_name: str) -> Callable[[Any], Any]:
        """Register a decorated artifact under one name.

        Args:
            artifact_name: The name of the artifact.

        Returns:
            A decorator storing the decorated object and returning it
            unchanged.
        """

        def decorator(artifact: Any) -> Any:
            self._dict[artifact_name] = artifact
            return artifact

        return decorator

    def names(self) -> tuple[str, ...]:
        """Return the registered component names in registration order."""
        return tuple(self._dict)

    def fetch(self, name_or_name_list: str | list[str]) -> Any:
        """Retrieve one registered artifact or a mapping of several.

        Args:
            name_or_name_list: A single registered name, or a list of names.

        Returns:
            The registered artifact for a string input; a mapping from each
            recognized name to its artifact for a list input.

        Raises:
            KeyError: If a requested single name is not registered.
            ValueError: If a list input matches no registered name.
        """
        if isinstance(name_or_name_list, list):
            artifacts = {}
            for name in name_or_name_list:
                if name in self._dict:
                    artifacts[name] = self._dict[name]
                else:
                    warnings.warn(f"Unregistered artifact_name: '{name}'. Ignoring this artifact.")
            if len(artifacts) == 0:
                raise ValueError("No registered artifacts found.")
            return artifacts
        if name_or_name_list in self._dict:
            return self._dict[name_or_name_list]
        raise KeyError(f"Unregistered artifact_name: '{name_or_name_list}'.")
