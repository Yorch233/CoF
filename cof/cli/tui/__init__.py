"""Reusable Rich terminal UI components for the CoF CLI.

Re-export the shared console, themed help rendering, selection menu, panels,
prompts, phase displays, and logo so CLI modules depend on this single façade
instead of the individual component modules.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from cof.cli.tui.console import get_console
from cof.cli.tui.display import StatusPhase, print_config_summary, print_help_table, print_phase, render_phase
from cof.cli.tui.help import ThemedTyperGroup, print_help
from cof.cli.tui.logo import build_logo
from cof.cli.tui.menu import SelectionMenu
from cof.cli.tui.panels import make_kv_panel, make_panel
from cof.cli.tui.prompts import prompt_number, prompt_text, prompt_validated_text, prompt_yes_no, rich_select

__all__ = [
    "SelectionMenu",
    "StatusPhase",
    "ThemedTyperGroup",
    "build_logo",
    "get_console",
    "make_kv_panel",
    "make_panel",
    "print_config_summary",
    "print_help_table",
    "print_help",
    "print_phase",
    "prompt_number",
    "prompt_text",
    "prompt_validated_text",
    "prompt_yes_no",
    "render_phase",
    "rich_select",
]
