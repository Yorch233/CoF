"""Publish complete files with an atomic replacement on the destination filesystem."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path and publish it only after its writer succeeds.

    A private subdirectory preserves the file extension for audio encoders
    without exposing unfinished WAVs to result-directory scans.

    Args:
        path: Final destination, replaced only on successful context exit.

    Yields:
        A temporary path on the destination filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".cof-write-") as directory:
        temporary = Path(directory) / path.name
        yield temporary
        temporary.replace(path)
