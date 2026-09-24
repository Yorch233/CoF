"""Build the shared rich progress display for inference and metric loops.

Every long-running workflow loop renders progress through these helpers so
CLI output stays visually consistent regardless of which stage is running.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def build_workflow_progress() -> Progress:
    """Build a uniformly styled progress bar for long-running workflow loops.

    Returns:
        A ``Progress`` instance with spinner, bar, count, and timing columns.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def track_workflow_items[T](
    items: Iterable[T],
    description: str,
    *,
    total: float | None = None,
    enabled: bool = True,
) -> Iterator[T]:
    """Yield items under a rich progress bar, or unchanged when the display is disabled.

    Args:
        items: Iterable to traverse.
        description: Progress-bar label.
        total: Expected item count; inferred when omitted.
        enabled: When false, yield items without any display.

    Yields:
        Items from the iterable, in order.
    """
    if not enabled:
        yield from items
        return
    with build_workflow_progress() as progress:
        yield from progress.track(items, total=total, description=description)
