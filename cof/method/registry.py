"""Independent pretraining and posttraining method extension points."""

from cof.utils.register import Register

PretrainingRegister = Register()
PostTrainingRegister = Register()


def load_builtin_methods() -> None:
    """Import built-in factories lazily without loading the Lightning framework."""
    from importlib import import_module

    import_module("cof.method.pretrain.base")
    import_module("cof.method.posttrain.cof.method")
