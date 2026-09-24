"""Load audio files or preprocessed tensor archives as datasets.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import os
from glob import glob
from typing import Any

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class AudioFolder(Dataset):
    """Audio dataset loading WAV files or preprocessed .pt files."""

    def __init__(
        self,
        audio_path: str | os.PathLike[str],
        sample_rate: int = 16000,
        return_path: bool = False,
        reverse: bool = False,
        lazy: bool | None = None,
    ) -> None:
        """Index audio files and load directory-backed WAVs on demand.

        ``lazy`` is retained as a read-only compatibility argument. Directory
        inputs are always path-backed so dataset construction never reads the
        complete audio collection into memory.

        Args:
            audio_path: Directory of WAV files or a ``.pt`` archive holding
                ``audio`` and ``sample_rate`` entries.
            sample_rate: Sample rate audio is resampled to on load.
            return_path: Whether ``__getitem__`` also returns the file stem.
            reverse: Whether to reverse the file or sample order.
            lazy: Unused compatibility argument; accepted and ignored.

        Raises:
            AssertionError: If the path does not exist or a ``.pt`` archive's
                sample rate disagrees with the requested one.
            NotImplementedError: If the path is neither a directory nor a
                ``.pt`` file.
        """
        assert os.path.exists(audio_path), f"Path not found: {audio_path}"

        self.reverse = reverse
        self.return_path = return_path
        del lazy

        self.audios = []
        self.audio_paths = []
        if os.path.isdir(audio_path):
            self.sample_rate = sample_rate
            self.audio_paths = sorted(glob(f"{audio_path}/*.wav"))
        elif audio_path.endswith(".pt"):
            data = torch.load(audio_path, weights_only=True)
            self.sample_rate = data["sample_rate"]
            assert self.sample_rate == sample_rate, "Sample rate mismatch"
            self.audios = list(data["audio"])
        else:
            raise NotImplementedError("Unsupported file type")

        if self.reverse:
            (self.audio_paths if self.audio_paths else self.audios).reverse()

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            Number of indexed files or in-memory samples.
        """
        return len(self.audio_paths) if self.audio_paths else len(self.audios)

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, str]:
        """Get audio data by index.

        Args:
            index: Position in the file or sample list.

        Returns:
            The waveform, plus the file stem when ``return_path`` is set.
        """
        if self.audio_paths:
            audio = self._load_audio(self.audio_paths[index])
            return (audio, os.path.basename(self.audio_paths[index])[:-4]) if self.return_path else audio
        return self.audios[index]

    def _load_audio(self, path: str) -> Tensor:
        """Load one WAV file and resample it to the configured rate.

        Args:
            path: WAV file path.

        Returns:
            The waveform tensor at the configured sample rate.
        """
        audio, source_sample_rate = torchaudio.load(path)
        if source_sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_sample_rate, self.sample_rate)
        return audio

    def get_data_loader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        reverse: bool | None = None,
    ) -> DataLoader[Any]:
        """Create dataloader with optional reverse option.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle the order each epoch.
            num_workers: Dataloader worker processes.
            pin_memory: Whether batches are pinned for CUDA transfer.
            reverse: When given, reverses the dataset order before loading.

        Returns:
            A dataloader over this dataset.
        """
        if reverse is not None:
            self.reverse = reverse
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
        )
