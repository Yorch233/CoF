"""Provide the shared Rich console for all CoF terminal output.

Instantiate one process-wide console bound to the CoF theme so every prompt,
table, and panel shares the same styling and TTY detection. Import
:func:`get_console` instead of creating additional consoles; a second console
would desynchronize cursor save/restore tricks used by the prompts.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import sys

from rich.console import Console

from cof.cli.tui.theme import COF_THEME

_CONSOLE = Console(theme=COF_THEME, force_terminal=sys.stdout.isatty())


def get_console() -> Console:
    """Return the process-wide themed console."""
    return _CONSOLE
