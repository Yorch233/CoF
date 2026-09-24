"""Backbone registry and automatic implementation discovery.

Importing this package walks every submodule below ``cof.backbone`` and
imports each ``modeling_*.py`` module, so backbone classes register themselves
into :data:`BackboneRegister` purely by being shipped under this package; no
central list of backbones has to be maintained.

Author: Qing Yao
Date: 2026/9/23
"""

from cof.backbone.registry import BackboneRegister
from cof.utils.discovery import import_modeling_modules as _import_modeling_modules

_import_modeling_modules(__name__, __path__)
del _import_modeling_modules

__all__ = ["BackboneRegister"]
