"""NCSN++ backbone package.

Importing :mod:`cof.backbone` discovers ``modeling_ncsnpp`` below this
package, which registers the ``ncsnpp_base`` backbone with the shared
registry; this package itself only re-exports the registry handle.

Author: Qing Yao
Date: 2026/9/23
"""

from cof.backbone.registry import BackboneRegister

__all__ = ["BackboneRegister"]
