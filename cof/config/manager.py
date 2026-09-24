"""Provide YAML-backed dynamic configuration helpers.

The helpers implement the tracked-preset inheritance model: a YAML file may
name ``inherit`` parents that are merged recursively before its own keys,
dictionaries merge key-by-key, and the resulting mapping is exposed either as
a plain ordered mapping or as the attribute-access ``Config`` wrapper that the
training, inference, and metric code reads everywhere.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import inspect
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml
from tabulate import tabulate

ConfigData = dict[str, Any]
ConfigClass = TypeVar("ConfigClass", bound=type[Any])


def _represent_ordered_dict(dumper: yaml.Dumper, data: OrderedDict[str, Any]) -> yaml.Node:
    """Render an ``OrderedDict`` as a plain YAML mapping, preserving key order."""
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.add_representer(OrderedDict, _represent_ordered_dict)


class BaseConfiguer:
    """Load and dump inherited YAML configuration files."""

    @classmethod
    def load(cls, file_path: str | os.PathLike[str]) -> OrderedDict[str, Any]:
        """Load YAML while recursively resolving relative ``inherit`` entries.

        Args:
            file_path: YAML file to load; ``inherit`` parents are resolved
                relative to the file that names them.

        Returns:
            The merged mapping in inheritance-then-definition order.

        Raises:
            ValueError: If the path does not point to a YAML file.
            TypeError: If the document root or an ``inherit`` field has the
                wrong shape.
        """
        path = Path(file_path)
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError("file_path must point to a YAML file")

        with path.open(encoding="utf-8") as stream:
            origin = yaml.safe_load(stream) or {}
        if not isinstance(origin, dict):
            raise TypeError(f"Configuration root in {path} must be a mapping")

        merged: OrderedDict[str, Any] = OrderedDict()
        inherited = origin.get("inherit")
        if inherited is not None:
            if not isinstance(inherited, (str, list)):
                raise TypeError(f"The inherit field in {path} must be a string or list")
            inherited_paths = [inherited] if isinstance(inherited, str) else inherited
            for inherited_path in inherited_paths:
                resolved = Path(inherited_path)
                if not resolved.is_absolute():
                    resolved = path.parent / resolved
                merged.update(cls.load(resolved))

        for key, value in origin.items():
            if key == "inherit":
                continue
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._merge_mapping(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> OrderedDict[str, Any]:
        """Recursively overlay one mapping onto another, shallow-copying leaves.

        Args:
            base: Lower-precedence mapping providing defaults.
            override: Higher-precedence mapping whose values win.

        Returns:
            The merged mapping; nested mappings merge key-by-key, everything
            else is replaced.
        """
        merged = OrderedDict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = BaseConfiguer._merge_mapping(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def dump(cls, data: Mapping[str, Any], output_path: str | os.PathLike[str]) -> None:
        """Write configuration data as YAML.

        Args:
            data: Mapping to serialize with insertion order preserved.
            output_path: Destination file (created or truncated).
        """
        del cls
        with Path(output_path).open("w", encoding="utf-8") as stream:
            yaml.dump(dict(data), stream, default_flow_style=False, sort_keys=False)


def read_yml(yml_path: str | os.PathLike[str]) -> OrderedDict[str, Any]:
    """Read an inherited YAML mapping.

    Args:
        yml_path: YAML file path.

    Returns:
        The merged mapping including resolved ``inherit`` parents.
    """
    return BaseConfiguer.load(yml_path)


def read_config_from_yaml(config_path: str | os.PathLike[str]) -> Config:
    """Read a YAML file, or ``config.yml`` within a run directory.

    Args:
        config_path: YAML file path or run directory.

    Returns:
        The configuration wrapper around the parsed mapping.

    Raises:
        ValueError: If the path is not a YAML file or does not exist.
    """
    path = Path(config_path)
    if path.is_dir():
        path = path / "config.yml"
    if path.suffix not in {".yml", ".yaml"}:
        raise ValueError(f"config_path must point to a YAML file, not {path!s}")
    if not path.exists():
        raise ValueError(f"The config file {path} does not exist")
    return Config(read_yml(path))


class Config:
    """Attribute-access wrapper around a dynamic, nested configuration mapping.

    Nested mappings are wrapped recursively, so ``config.data.sample_rate``
    works alongside dotted-path lookups (``config.get("data.sample_rate")``).
    ``update`` accepts dotted keys and creates intermediate groups, which lets
    callers write into the hierarchy without rebuilding dictionaries.
    """

    _MAX_LENGTH = 100

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize configuration attributes from a mapping.

        Args:
            config: Mapping whose keys become attributes; nested mappings are
                wrapped recursively and dotted keys create nested groups.
        """
        self.update(config)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        """Wrap nested mappings as Config objects, leaving other values intact."""
        if isinstance(value, Config) or not isinstance(value, Mapping):
            return value
        return Config(value)

    @staticmethod
    def _unwrap(value: Any) -> Any:
        """Convert wrapped Config objects back to plain nested dictionaries."""
        if isinstance(value, Config):
            return value.dict()
        if isinstance(value, dict):
            return {key: Config._unwrap(item) for key, item in value.items()}
        return value

    def dict(self) -> ConfigData:
        """Return public configuration items as a nested plain dictionary.

        Returns:
            A deep plain-dictionary copy with private keys removed.
        """
        return {key: self._unwrap(value) for key, value in self.__dict__.items() if not key.startswith("_")}

    def get(self, field: str, default: Any = None) -> Any:
        """Get a configuration value with a fallback, supporting dotted paths.

        Args:
            field: Configuration key, or a dotted path such as
                ``"data.sample_rate"``.
            default: Value returned when any path segment is absent.

        Returns:
            The configured value or the default.
        """
        current: Any = self
        for part in field.split("."):
            if isinstance(current, Config):
                current = getattr(current, part, default)
            elif isinstance(current, dict):
                current = current.get(part, default)
            else:
                return default
            if current is default:
                return default
        return current

    @staticmethod
    def unwrap(value: Any) -> Any:
        """Alias of :meth:`_unwrap` for external coercion of nested groups."""
        return Config._unwrap(value)

    def update(self, config: Mapping[str, Any]) -> None:
        """Update configuration attributes, nesting values under dotted keys.

        Args:
            config: Mapping whose keys overwrite current attributes; dotted
                keys create or extend nested groups.
        """
        for key, value in config.items():
            target: Config = self
            parts = key.split(".")
            for part in parts[:-1]:
                nested = getattr(target, part, None)
                if not isinstance(nested, Config):
                    nested = Config({})
                    target.__dict__[part] = nested
                target = nested
            target.__dict__[parts[-1]] = self._wrap(value)

    def save(self, save_path: str | os.PathLike[str] | None = None, file_name: str = "config.yml") -> None:
        """Save configuration to a run directory.

        Args:
            save_path: Destination directory; defaults to ``run_path``.
            file_name: Output file name inside the directory.
        """
        destination = Path(save_path if save_path is not None else self.run_path)
        destination.mkdir(parents=True, exist_ok=True)
        BaseConfiguer.dump(self.dict(), destination / file_name)

    def print(self) -> None:
        """Print configuration as a compact table."""
        rows = [[str(key), self._truncate(str(value))] for key, value in self.dict().items()]
        print("Configuration:")
        print(tabulate(rows, headers=["Param", "Value"], tablefmt="pretty"))

    def _truncate(self, sentence: str) -> str:
        """Shorten values past the display limit with an ellipsis.

        Args:
            sentence: Rendered value text.

        Returns:
            The original text, or a truncated prefix ending in ``...``.
        """
        if len(sentence) <= self._MAX_LENGTH:
            return sentence
        return f"{sentence[: self._MAX_LENGTH - 4]}..."


def config_from_yaml(config_path: str, key_value: str | None = None) -> Callable[[ConfigClass], ConfigClass]:
    """Inject YAML values as constructor defaults for a decorated class.

    Args:
        config_path: YAML file supplying default values.
        key_value: Optional top-level key selecting a sub-mapping.

    Returns:
        A class decorator that merges YAML defaults, YAML-parsed values, and
        explicit keyword arguments into ``__init__``.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """

    def decorator(cls: ConfigClass) -> ConfigClass:
        """Capture YAML defaults and wrap the decorated constructor."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file {config_path} not found")
        configs: Mapping[str, Any] = read_yml(path)
        if key_value is not None:
            configs = configs[key_value]

        signature = inspect.signature(cls.__init__)
        parameters = signature.parameters
        defaults = {
            name: _parse_config_value(parameter.annotation, configs[name])
            for name, parameter in parameters.items()
            if name != "self" and name in configs
        }
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            """Merge declared defaults, YAML values and explicit constructor arguments."""
            merged = {
                name: parameter.default
                for name, parameter in parameters.items()
                if parameter.default is not inspect.Parameter.empty and name != "self"
            }
            merged.update(defaults)
            merged.update(kwargs)
            missing = [
                name
                for name, parameter in parameters.items()
                if name != "self" and parameter.default is inspect.Parameter.empty and name not in merged
            ]
            if missing:
                raise ValueError(f"Missing required parameters: {missing}")
            original_init(self, *args, **merged)

        cls.__init__ = new_init
        return cls

    return decorator


def _parse_config_value(expected_type: Any, value: Any) -> Any:
    """Convert a YAML value to its annotated constructor type.

    Args:
        expected_type: Annotation from the decorated constructor; ``Any`` and
            missing annotations pass the value through unchanged.
        value: Parsed YAML value.

    Returns:
        The value coerced to ``expected_type`` when the annotation is a
        callable type, or ``None`` for a ``None`` value.
    """
    if expected_type in {Any, inspect.Parameter.empty}:
        return value
    if expected_type is torch.device:
        return torch.device(value)
    if inspect.isclass(expected_type) and issubclass(expected_type, dict):
        return dict(value)
    return expected_type(value) if value is not None else None
