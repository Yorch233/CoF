"""Instantiate the shared backbone extension registry.

Backbone modules import :data:`BackboneRegister` and decorate their classes
with it at import time; the automatic discovery in :mod:`cof.backbone` relies
on that side effect, so this module must stay the single registry instance.

Author: Qing Yao
Date: 2026/9/23
"""

from cof.utils.register import Register

BackboneRegister = Register()
