"""Instance-local warped STFT with independent configuration and device caches."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from cof.config.manager import Config


def get_window(window_type: str, window_length: int) -> Tensor:
    """Create a Hann or square-root Hann window.

    Args:
        window_type: ``"sqrthann"`` or ``"hann"``.
        window_length: Number of window taps.

    Returns:
        The periodic window tensor.

    Raises:
        NotImplementedError: If the window type is not implemented.
    """
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    if window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    raise NotImplementedError(f"Window type {window_type!r} is not implemented")


class StftTransform:
    """Apply the magnitude-warped STFT used by the experiment data path."""

    def __init__(
        self,
        n_fft: int = 510,
        num_frames: int = 256,
        hop_length: int = 128,
        spec_abs_exponent: float = 0.5,
        spec_factor: float = 0.33,
        window: str = "sqrthann",
    ) -> None:
        """Initialize transform parameters and reset device-window state.

        Args:
            n_fft: FFT size.
            num_frames: Configured frame count for dataset geometry.
            hop_length: Hop between frames.
            spec_abs_exponent: Magnitude-warping exponent.
            spec_factor: Magnitude-warping scale.
            window: Window type name.
        """
        if n_fft < 2 or num_frames < 1 or not 0 < hop_length <= n_fft:
            raise ValueError("STFT requires n_fft >= 2, num_frames >= 1 and 0 < hop_length <= n_fft")
        if (
            not math.isfinite(spec_abs_exponent)
            or spec_abs_exponent <= 0
            or not math.isfinite(spec_factor)
            or spec_factor <= 0
        ):
            raise ValueError("STFT warping exponent and scale must be finite and positive")
        self.n_fft = n_fft
        self.num_frames = num_frames
        self.hop_length = hop_length
        self.spec_abs_exponent = spec_abs_exponent
        self.spec_factor = spec_factor
        self.window_type = window
        self.window = get_window(window, n_fft)
        self.windows: dict[torch.device, Tensor] = {}

    @classmethod
    def from_config(cls, config: Config) -> StftTransform:
        """Construct an independent transform from a run's data configuration."""
        defaults = {
            "n_fft": 510,
            "num_frames": 256,
            "hop_length": 128,
            "spec_abs_exponent": 0.5,
            "spec_factor": 0.33,
            "window": "sqrthann",
        }
        return cls(**{name: config.get(f"data.{name}", value) for name, value in defaults.items()})

    def to_config(self) -> dict[str, object]:
        """Return portable transform parameters without cached window tensors."""
        return {
            "n_fft": self.n_fft,
            "num_frames": self.num_frames,
            "hop_length": self.hop_length,
            "spec_abs_exponent": self.spec_abs_exponent,
            "spec_factor": self.spec_factor,
            "window": self.window_type,
        }

    def _get_window(self, tensor: Tensor) -> Tensor:
        """Return a cached transform window on the tensor device.

        Args:
            tensor: Tensor whose device the window is moved to.

        Returns:
            The window tensor on the requested device.
        """
        if self.window is None:
            raise RuntimeError("STFT window was not initialized")
        if tensor.device not in self.windows:
            self.windows[tensor.device] = self.window.to(tensor.device)
        return self.windows[tensor.device]

    def stft(self, audio: Tensor, transform: bool = True) -> Tensor:
        """Convert audio to a complex spectrum.

        Args:
            audio: Waveform tensor ``(..., time)``.
            transform: Apply magnitude warping after the STFT.

        Returns:
            The (optionally warped) complex spectrum.
        """
        if self.n_fft is None or self.hop_length is None:
            raise RuntimeError("STFT parameters were not initialized")
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._get_window(audio),
            center=True,
            return_complex=True,
        )
        return self.magnitude_warping(spectrum) if transform else spectrum

    def istft(self, spectrum: Tensor, transform: bool = True, length: int | None = None) -> Tensor:
        """Convert a complex spectrum back to audio.

        Args:
            spectrum: Complex spectrum ``(..., freq, frames)``.
            transform: Invert the magnitude warping before the ISTFT.
            length: Output waveform length to trim or pad to.

        Returns:
            The reconstructed waveform.
        """
        if self.n_fft is None or self.hop_length is None:
            raise RuntimeError("STFT parameters were not initialized")
        restored = self.invert_magnitude_warping(spectrum) if transform else spectrum
        return torch.istft(
            restored,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._get_window(restored),
            center=True,
            length=length,
        )

    def magnitude_warping(self, spectrum: Tensor) -> Tensor:
        """Compress spectrum magnitudes while retaining phase.

        Args:
            spectrum: Complex spectrum tensor.

        Returns:
            The magnitude-warped spectrum; a unit exponent leaves magnitudes
            untouched.

        Raises:
            RuntimeError: If the warping parameters were not initialized.
        """
        if self.spec_abs_exponent is None or self.spec_factor is None:
            raise RuntimeError("Magnitude-warping parameters were not initialized")
        if self.spec_abs_exponent != 1:
            spectrum = spectrum.abs().pow(self.spec_abs_exponent) * torch.exp(1j * spectrum.angle())
        return spectrum * self.spec_factor

    def invert_magnitude_warping(self, spectrum: Tensor) -> Tensor:
        """Undo spectrum magnitude compression.

        Args:
            spectrum: Magnitude-warped complex spectrum.

        Returns:
            The unwarped spectrum.

        Raises:
            RuntimeError: If the warping parameters were not initialized.
        """
        if self.spec_abs_exponent is None or self.spec_factor is None:
            raise RuntimeError("Magnitude-warping parameters were not initialized")
        spectrum = spectrum / self.spec_factor
        if self.spec_abs_exponent != 1:
            spectrum = spectrum.abs().pow(1 / self.spec_abs_exponent) * torch.exp(1j * spectrum.angle())
        return spectrum

    def to_stft(
        self,
        audio: Tensor,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        """Normalize and transform audio, returning an inverse closure.

        Args:
            audio: Waveform tensor ``(..., time)``.
            device: Device the spectrum is computed on.

        Returns:
            The padded, warped spectrum and a closure mapping any spectrum
            estimate back to unnormalized audio on CPU.

        Raises:
            ValueError: If the input audio is empty.
        """
        if audio.numel() == 0:
            raise ValueError("Cannot enhance empty audio")
        audio_length = audio.size(-1)
        audio = audio.reshape(1, -1)
        normalization = max(float(audio.abs().max()), torch.finfo(audio.dtype).eps)
        normalized = audio.to(device) / normalization
        minimum_length = self.n_fft // 2 + 1
        if normalized.size(-1) < minimum_length:
            # A full window also keeps padded spectral frames beyond the original audio.
            normalized = F.pad(normalized, (0, self.n_fft - normalized.size(-1)))
        spectrum = pad_spec(self.stft(normalized).unsqueeze(0))

        def invert(value: Tensor) -> Tensor:
            """Restore the input length and scale using this transform instance."""
            restored = self.istft(value.squeeze(), length=audio_length)
            return restored.squeeze().cpu() * normalization

        return spectrum, invert


def pad_spec(spectrum: Tensor) -> Tensor:
    """Pad the time axis to a multiple of 64.

    Args:
        spectrum: Spectrum tensor whose last axis holds frames.

    Returns:
        The zero-padded spectrum.
    """
    frames = spectrum.size(3)
    padding = 64 - frames % 64 if frames % 64 else 0
    return torch.nn.ZeroPad2d((0, padding, 0, 0))(spectrum)
