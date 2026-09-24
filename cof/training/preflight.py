"""Select NCSN++ operators and confirm training with the operator.

The preflight is the last interactive gate before a training run: it probes
the NCSN++ JIT compilation requirement, records the outcome in the run
configuration, prints the effective parameters, and asks for confirmation.
Every decline raises ``TrainingCancelled`` so no half-configured run directory
is left behind.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from cof.backbone.ncsnpp.ncsnpp_utils.op.backend import (
    CUDA_JIT_BACKEND,
    PYTORCH_NATIVE_BACKEND,
    enable_cuda_jit,
    use_pytorch_native,
)
from cof.backbone.registry import BackboneRegister
from cof.config.manager import Config

BACKEND_FIELD = "host.ncsnpp_operator_backend"
JIT_STATUS_FIELD = "host.ncsnpp_cuda_jit_status"


class TrainingCancelled(RuntimeError):
    """Raised when the user declines a training preflight confirmation."""


def prepare_training_launch(config: Config, *, assume_yes: bool = False) -> None:
    """Select NCSN++ operators, show final parameters, and confirm training.

    Args:
        config: Effective run configuration; updated in place with the
            resolved operator backend, JIT status, and preflight status.
        assume_yes: Skip the interactive confirmation (attended review of the
            printed configuration, or a genuinely unattended job).

    Raises:
        TrainingCancelled: If a backbone requirement fails or the user (or
            policy, with ``require_cuda_jit``) declines the launch.
        ValueError: If the configured backbone or operator backend is
            unsupported.
    """
    from cof.cli.tui import get_console, print_config_summary, prompt_yes_no

    backbone = str(config.get("model.backbone", "ncsnpp_base") or "ncsnpp_base")
    if not backbone.startswith("ncsnpp"):
        component = BackboneRegister.fetch(backbone)
        hook = getattr(component, "training_preflight", None)
        if hook is not None:
            hook(config)
        config.update({BACKEND_FIELD: "not_applicable", JIT_STATUS_FIELD: "not_applicable"})
        print_config_summary(config.dict(), title="Training parameters", boxed=False)
        if not assume_yes and not prompt_yes_no("Start training with these parameters?", default=True):
            raise TrainingCancelled("Training was declined")
        return

    requested_backend = config.get(BACKEND_FIELD, CUDA_JIT_BACKEND)
    require_cuda_jit = bool(config.get("host.require_cuda_jit", False))
    if requested_backend == PYTORCH_NATIVE_BACKEND:
        if require_cuda_jit:
            raise TrainingCancelled("CUDA JIT is required but the configured backend is PyTorch native")
        use_pytorch_native()
        config.update(
            {
                BACKEND_FIELD: PYTORCH_NATIVE_BACKEND,
                JIT_STATUS_FIELD: config.get(JIT_STATUS_FIELD, "not_attempted"),
            }
        )
    elif requested_backend == CUDA_JIT_BACKEND:
        try:
            enable_cuda_jit()
        except Exception as error:
            config.update({JIT_STATUS_FIELD: "failed"})
            get_console().print(f"[error]CUDA JIT initialization failed:[/error] {error}")
            if require_cuda_jit:
                raise TrainingCancelled("CUDA JIT is required and initialization failed") from error
            if not assume_yes and not prompt_yes_no(
                "Continue training with native PyTorch NCSN++ operators?",
                default=True,
            ):
                raise TrainingCancelled("Native PyTorch operator fallback was declined") from error
            use_pytorch_native()
            config.update({BACKEND_FIELD: PYTORCH_NATIVE_BACKEND})
        else:
            config.update({BACKEND_FIELD: CUDA_JIT_BACKEND, JIT_STATUS_FIELD: "available"})
    else:
        raise ValueError(f"Unsupported NCSN++ operator backend: {requested_backend!r}")

    print_config_summary(config.dict(), title="Training parameters", boxed=False)
    if not assume_yes and not prompt_yes_no("Start training with these parameters?", default=True):
        raise TrainingCancelled("Training was declined")
