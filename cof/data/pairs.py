"""Validate the shared clean/noisy filename contract before indexing audio."""

from pathlib import Path

SPLITS = ("train", "valid", "test")


def paired_wavs(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Return exactly aligned clean/noisy WAV pairs for a dataset split.

    Args:
        dataset_root: Registered dataset root directory.
        split: Dataset split to pair.

    Returns:
        Clean/noisy path pairs sorted by the shared filename.

    Raises:
        ValueError: If the split is unsupported, either side is empty, or the
            clean and noisy filename sets differ.
    """
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}; choose from {SPLITS}")
    clean_dir = dataset_root / split / "clean"
    noisy_dir = dataset_root / split / "noisy"
    clean = {path.name: path for path in clean_dir.glob("*.wav")}
    noisy = {path.name: path for path in noisy_dir.glob("*.wav")}
    if not clean or not noisy:
        raise ValueError(f"Dataset split has no paired WAV files: {dataset_root / split}")
    if clean.keys() != noisy.keys():
        missing_noisy = sorted(clean.keys() - noisy.keys())
        missing_clean = sorted(noisy.keys() - clean.keys())
        raise ValueError(f"Unpaired WAV files; missing noisy={missing_noisy}, missing clean={missing_clean}")
    return [(clean[name], noisy[name]) for name in sorted(clean)]
