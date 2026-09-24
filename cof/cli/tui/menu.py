"""Drive an arrow-key terminal selection menu for CoF commands.

Put a POSIX terminal into cbreak mode, redraw a numbered choice list in place,
and fall back to numbered line input when no TTY is available. The
interaction model is based on Hugging Face Accelerate's terminal menu:
https://github.com/huggingface/accelerate/tree/main/src/accelerate/commands/menu/

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import os
import select
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from rich.text import Text

from cof.cli.tui.console import get_console


def as_question(prompt: str) -> str:
    """Normalize an interactive prompt as a question."""
    normalized = prompt.strip()
    return normalized if normalized.endswith(("?", "？")) else f"{normalized}?"


@contextmanager
def _raw_terminal() -> Iterator[None]:
    """Temporarily put a POSIX terminal into cbreak input mode."""
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        # NOTE: cbreak (unlike raw) keeps output processing enabled, so newlines remain
        # aligned at column zero while arrow keys arrive without waiting for Enter.
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _read_character(timeout: float | None = None) -> str | None:
    """Read one byte from stdin, waiting at most ``timeout`` seconds.

    Returns:
        The decoded character, or ``None`` when the timeout expired.

    Raises:
        EOFError: If stdin reached end of input.
    """
    descriptor = sys.stdin.fileno()
    if timeout is not None:
        readable, _, _ = select.select([descriptor], [], [], timeout)
        if not readable:
            return None
    character = os.read(descriptor, 1)
    if not character:
        raise EOFError
    return character.decode(errors="ignore")


def _read_key() -> str:
    """Read one key, translating common ANSI arrow sequences.

    Returns:
        ``"up"``, ``"down"``, ``"escape"``, or the raw character.
    """
    character = _read_character()
    if character == "\x03":
        raise KeyboardInterrupt
    if character != "\x1b":
        assert character is not None
        return character
    prefix = _read_character(0.05)
    if prefix != "[":
        return "escape"
    suffix = _read_character(0.05)
    return {"A": "up", "B": "down"}.get(suffix or "", "escape")


class SelectionMenu:
    """Render a keyboard-driven, single-choice terminal menu."""

    def __init__(self, prompt: str, choices: list[str]) -> None:
        """Initialize a menu with a prompt and at least one choice.

        Args:
            prompt: Question rendered above the choices.
            choices: Non-empty list of selectable entries.

        Raises:
            ValueError: If ``choices`` is empty.
        """
        if not choices:
            raise ValueError("SelectionMenu requires at least one choice")
        self.prompt = as_question(prompt)
        self.choices = choices

    def _render(self, position: int, *, show_hint: bool = True, default: int | None = None) -> Text:
        """Render the prompt block with the current position highlighted.

        Args:
            position: Zero-based index of the highlighted choice.
            show_hint: Append the interactive key hint line.
            default: Default index named by the hint lines.

        Returns:
            The rendered Rich text block.
        """
        rendered = Text.assemble(("◇ ", "accent"), (self.prompt, "bold text"))
        for index, choice in enumerate(self.choices):
            rendered.append("\n│  ", style="dim")
            rendered.append(f"{index + 1}. ", style="dim")
            if index == position:
                rendered.append("● ", style="accent")
                rendered.append(choice, style="bold accent")
            else:
                rendered.append("○ ", style="dim")
                rendered.append(choice, style="text")
        if show_hint:
            default_position = position if default is None else default
            rendered.append(
                f"\n│  Use ↑/↓ or a number, then press Enter. Default: {self.choices[default_position]}",
                style="dim",
            )
        elif default is not None:
            rendered.append(f"\n│  Enter an option number. Default: {self.choices[default]}", style="dim")
        return rendered

    def _print_result(self, position: int) -> None:
        """Print the collapsed question-and-answer block after selection."""
        result = Text.assemble(
            ("◇ ", "success"),
            (self.prompt, "text"),
            ("\n│  Selected: ", "dim"),
            (self.choices[position], "accent"),
            ("\n│", "dim"),
        )
        get_console().print(result)

    def _fallback(self, default: int) -> int:
        """Use numbered line input when raw terminal control is unavailable.

        Args:
            default: Zero-based index returned when the input is empty.

        Returns:
            The zero-based index of the chosen entry.
        """
        console = get_console()
        console.print(self._render(default, show_hint=False, default=default))
        while True:
            try:
                value = console.input(
                    f"[dim]│  Enter a number from 1 to {len(self.choices)} (default: {default + 1}): [/dim]"
                ).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]│[/dim] [error]Configuration cancelled.[/error]")
                raise SystemExit(1) from None
            if not sys.stdin.isatty():
                console.print()
                console.print("[dim]│[/dim]")
            if not value:
                return default
            if value.isdigit() and 1 <= int(value) <= len(self.choices):
                return int(value) - 1
            console.print(f"[dim]│[/dim] [error]Enter a number from 1 to {len(self.choices)}.[/error]")

    def run(self, default: int = 0) -> int:
        """Return the selected zero-based choice index.

        Args:
            default: Zero-based index to preselect.

        Returns:
            The zero-based index of the chosen entry.

        Raises:
            ValueError: If ``default`` is outside the choice range.
            KeyboardInterrupt: If the user presses Escape or Ctrl+C.
        """
        if not 0 <= default < len(self.choices):
            raise ValueError("default selection is outside the choice range")
        if os.name != "posix" or not sys.stdin.isatty():
            # NOTE: Raw terminal control needs a POSIX TTY; otherwise read numbered lines.
            position = self._fallback(default)
            self._print_result(position)
            return position

        console = get_console()
        position = default

        def redraw() -> None:
            # NOTE: Move up one line per choice plus the prompt line, then clear
            # to end of screen so the next render replaces the previous block.
            console.file.write(f"\x1b[{len(self.choices) + 2}A\x1b[J")
            console.file.flush()
            console.print(self._render(position, default=default))

        with _raw_terminal():
            console.print(self._render(position, default=default))
            while True:
                key = _read_key()
                if key == "up":
                    position = (position - 1) % len(self.choices)
                    redraw()
                elif key == "down":
                    position = (position + 1) % len(self.choices)
                    redraw()
                elif key in {"\r", "\n"}:
                    break
                elif key == "escape":
                    raise KeyboardInterrupt
                elif key.isdigit() and 1 <= int(key) <= len(self.choices):
                    position = int(key) - 1
                    redraw()
        self._print_result(position)
        return position
