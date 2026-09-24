"""Build the CoF logo renderable for terminal panels.

A single declarative builder over the theme's logo color stops; the glyphs
below are the only place the CoF wordmark is defined.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from rich.text import Text

from cof.cli.tui.theme import LOGO_BOTTOM, LOGO_MID, LOGO_TOP


def build_logo() -> Text:
    """Build the CoF terminal logo."""
    logo = Text()
    logo.append(" ██████╗  ██████╗ ███████╗\n", style=f"bold {LOGO_TOP}")
    logo.append("██╔════╝██╔═══██╗██╔════╝\n", style=f"bold {LOGO_TOP}")
    logo.append("██║     ██║   ██║█████╗  \n", style=f"bold {LOGO_MID}")
    logo.append("██║     ██║   ██║██╔══╝  \n", style=f"bold {LOGO_MID}")
    logo.append("╚█████╗ ╚██████╔╝██║     \n", style=f"bold {LOGO_BOTTOM}")
    logo.append(" ╚════╝  ╚═════╝ ╚═╝     ", style=f"bold {LOGO_BOTTOM}")
    return logo
