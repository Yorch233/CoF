"""Build aligned clean/noisy complex-spectrogram examples from paired folders.

The dataset pairs two ``AudioFolder`` instances by index, aligns and crops
the waveforms, and transforms them through the shared magnitude-warped STFT,
so no auxiliary spectrogram cache exists to fall out of sync with the audio.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from cof.config.manager import Config
from cof.data.audio_folder import AudioFolder
from cof.data.pairs import paired_wavs
from cof.data.stft import StftTransform


class ComplexSpecDataset(Dataset[tuple[Tensor, ...]]):
    """Load aligned clean and noisy spectra without an auxiliary cache."""

    def __init__(
        self,
        config: Config,
        dataset: str = "voicebank",
        subset: Literal["train", "valid", "test"] = "train",
        shuffle_spec: bool | None = None,
        normalize_audio: bool | None = None,
        return_raw: bool = False,
        return_spec: bool = True,
        dummy: bool = False,
    ) -> None:
        """Initialize aligned audio folders and spectrum transforms.

        Args:
            config: Run configuration carrying dataset paths, audio geometry,
                and STFT parameters.
            dataset: Registered dataset ID.
            subset: Split to load.
            shuffle_spec: Whether waveform crops start at a random offset
                during training; ``None`` disables random offsets.
            normalize_audio: Peak-normalization override; ``None`` reads the
                configured default.
            return_raw: Return unnormalized raw waveforms instead of
                spectra.
            return_spec: Return STFT spectra instead of waveforms.
            dummy: Serve a fixed 200-example length without touching audio
                (smoke-testing hook).

        Raises:
            ValueError: If the dataset ID is not configured or the split is empty or unpaired.
        """
        datasets = Config.unwrap(config.get("registry.datasets") or {})
        if dataset not in datasets:
            raise ValueError(f"Dataset {dataset!r} is not configured")
        self.sample_rate = int(config.get("data.sample_rate"))
        self.audio_length = int(config.get("data.audio_length"))
        self.data_dir = datasets[dataset]
        self.subset = subset
        self.spatial_channels = int(config.get("data.spatial_channels"))
        self.num_frames = int(config.get("data.num_frames"))
        self.hop_length = int(config.get("data.hop_length"))
        self.n_fft = int(config.get("data.n_fft"))
        self.spec_abs_exponent = float(config.get("data.spec_abs_exponent"))
        self.spec_factor = float(config.get("data.spec_factor"))
        self.window = config.get("data.window")
        paired_wavs(Path(self.data_dir), subset)
        self.clean_files = AudioFolder(
            audio_path=f"{self.data_dir}/{subset}/clean",
            sample_rate=self.sample_rate,
        )
        self.noisy_files = AudioFolder(
            audio_path=f"{self.data_dir}/{subset}/noisy",
            sample_rate=self.sample_rate,
        )
        self.shuffle_spec = bool(shuffle_spec)
        self.normalize_audio = config.get("data.normalize_audio", True) if normalize_audio is None else normalize_audio
        self.return_spec = return_spec
        self.return_raw = return_raw
        self.dummy = dummy
        self.transform = StftTransform(
            n_fft=self.n_fft,
            num_frames=self.num_frames,
            hop_length=self.hop_length,
            spec_abs_exponent=self.spec_abs_exponent,
            spec_factor=self.spec_factor,
            window=self.window,
        )

    def __len__(self) -> int:
        """Return the number of paired examples, or the dummy training size.

        Returns:
            200 in dummy mode, otherwise the paired file count.
        """
        return 200 if self.dummy else len(self.noisy_files)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        """Load, align, crop, normalize, and transform one paired example.

        Args:
            index: Pair position shared by both folders.

        Returns:
            Clean/noisy tensors — raw waveforms, normalized waveforms, or
            complex spectra, depending on the return flags.

        Raises:
            ValueError: If the audio has fewer channels than requested.
        """
        clean = self.clean_files[index]
        noisy = self.noisy_files[index]
        minimum_length = min(clean.size(-1), noisy.size(-1))
        clean, noisy = clean[..., :minimum_length], noisy[..., :minimum_length]
        if clean.ndim == 2 and self.spatial_channels == 1:
            clean, noisy = clean[0].unsqueeze(0), noisy[0].unsqueeze(0)
        if self.spatial_channels > clean.size(0):
            raise ValueError(
                f"Requested {self.spatial_channels} channels from audio containing {clean.size(0)} channels"
            )
        clean, noisy = clean[: self.spatial_channels], noisy[: self.spatial_channels]
        if self.return_raw:
            return clean, noisy

        normalization = noisy.abs().max().clamp_min(torch.finfo(noisy.dtype).eps)
        target_length = self.audio_length if not self.return_spec else (self.num_frames - 1) * self.hop_length
        clean, noisy = self._fit_length(clean, noisy, target_length=target_length)
        if self.normalize_audio:
            clean, noisy = clean / normalization, noisy / normalization
        if not self.return_spec:
            return clean, noisy
        return self.transform.stft(clean), self.transform.stft(noisy)

    def _fit_length(self, *signals: Tensor, target_length: int) -> tuple[Tensor, ...]:
        """Center-pad or randomly crop paired waveforms to the configured length.

        Args:
            *signals: Paired waveforms of equal length.
            target_length: Required sample count along the time axis.

        Returns:
            The signals, all padded or cropped to ``target_length``.
        """
        current_length = signals[0].size(-1)
        padding = max(target_length - current_length, 0)
        if padding:
            pad = (padding // 2, padding // 2 + padding % 2)
            return tuple(F.pad(signal, pad) for signal in signals)

        maximum_start = current_length - target_length
        start = int(np.random.uniform(0, maximum_start)) if self.shuffle_spec and maximum_start else maximum_start // 2
        stop = start + target_length
        return tuple(signal[..., start:stop] for signal in signals)
