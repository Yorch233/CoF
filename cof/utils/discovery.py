"""Import modules that populate registries through decorators.

Registered implementations are discovered by convention rather than by a
central list: importing every ``modeling_*.py`` module below a package is
what triggers each module's registry decorations, so a new implementation
only needs to live below the package.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from pkgutil import walk_packages


def import_modeling_modules(package_name: str, package_path: Iterable[str]) -> None:
    """Import every ``modeling_*.py`` module below a package.

    Args:
        package_name: Fully qualified package name forming the module prefix.
        package_path: Package search paths passed to ``walk_packages``.
    """
    prefix = f"{package_name}."
    for module_info in walk_packages(package_path, prefix):
        module_name = module_info.name.rsplit(".", 1)[-1]
        if module_name.startswith("modeling_"):
            import_module(module_info.name)
