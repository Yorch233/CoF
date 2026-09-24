"""Expose the ``cof`` command-line entry points for training, inference, and evaluation.

Keep this package a thin façade: each concrete command lives in its own module, and the
application wiring in :mod:`cof.cli.app` imports and registers the modules re-exported
through the subpackages below.

Author: Qing Yao
Date: 2026/9/23
"""
