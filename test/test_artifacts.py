"""Regression coverage for pairing, checkpoint evaluation and interrupted inference."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio
from torch import nn

from cof.backbone.registry import BackboneRegister
from cof.config.manager import read_config_from_yaml
from cof.data.complex_spec_dataset import ComplexSpecDataset
from cof.model import GenerativeModel4SE
from cof.pipeline.evaluation import evaluate_directory
from cof.pipeline.inference import enhance_audio_files, run_generative_inference
from cof.pipeline.provenance import file_sha256, read_json, write_json
from cof.utils.paths import PROJECT_ROOT


class ArtifactBackbone(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, state, time):
        return state[:, :1] * self.scale


@pytest.fixture
def artifact_workspace(tmp_path, monkeypatch):
    monkeypatch.setitem(BackboneRegister._dict, "artifact_test", ArtifactBackbone)
    dataset = tmp_path / "dataset"
    wave = np.sin(np.arange(4096) * 0.08).astype(np.float32) * 0.1
    for split in ("train", "valid", "test"):
        for kind in ("clean", "noisy"):
            folder = dataset / split / kind
            folder.mkdir(parents=True)
            for index in range(2):
                sf.write(folder / f"{index}.wav", wave + (0.01 if kind == "noisy" else 0), 16000)
    settings = {"dataset": "test", "datasets": {"test": str(dataset)}, "results_dir": str(tmp_path / "results")}
    monkeypatch.setattr("cof.pipeline.provenance._local_settings", lambda: settings)
    return tmp_path, dataset


@pytest.mark.parametrize(("clean_names", "noisy_names"), [(["a"], ["b"]), (["a", "b"], ["a"]), ([], [])])
def test_training_rejects_unpaired_split_before_audio_index(tmp_path, monkeypatch, clean_names, noisy_names):
    for kind, names in (("clean", clean_names), ("noisy", noisy_names)):
        folder = tmp_path / "train" / kind
        folder.mkdir(parents=True)
        for name in names:
            (folder / f"{name}.wav").touch()
    config = read_config_from_yaml(PROJECT_ROOT / "config/SB-VE/pretrain.yml")
    config.update({"registry.datasets": {"test": str(tmp_path)}})

    def unexpected_index(*args, **kwargs):
        pytest.fail("Audio indexing started before pair validation")

    monkeypatch.setattr("cof.data.complex_spec_dataset.AudioFolder", unexpected_index)
    with pytest.raises(ValueError, match="Unpaired|no paired WAV"):
        ComplexSpecDataset(config, dataset="test")


@pytest.mark.parametrize("mode", ["inference", "enhance"])
def test_first_run_interruption_resumes_only_missing_audio(artifact_workspace, monkeypatch, mode):
    root, dataset = artifact_workspace
    model = GenerativeModel4SE("SBVE", backbone="artifact_test")
    checkpoint = model.save_checkpoint(root / "model.safetensors")
    output = root / "results/model/SB_SDE_Solver_N=4" if mode == "inference" else root / "enhanced"
    manifest_name = "inference.json" if mode == "inference" else "enhance.json"

    def run():
        if mode == "inference":
            return run_generative_inference(checkpoint=checkpoint, device="cpu", progress=False)
        return enhance_audio_files(
            checkpoint=checkpoint,
            inputs=[dataset / "test/noisy"],
            output_dir=output,
            device="cpu",
            progress=False,
        )

    original_save = torchaudio.save
    writes = []

    def interrupted_save(path, *args, **kwargs):
        writes.append(Path(path).name)
        if len(writes) == 2:
            Path(path).write_bytes(b"partial audio")
            raise RuntimeError("simulated encoder interruption")
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(torchaudio, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated encoder interruption"):
        run()
    assert read_json(output / manifest_name)["status"] == "in_progress"
    assert read_json(output / manifest_name)["files"] == ["0.wav", "1.wav"]
    assert sorted(p.name for p in output.glob("*.wav")) == ["0.wav"]
    first_hash = file_sha256(output / "0.wav")
    if mode == "inference":
        with pytest.raises(ValueError, match="incomplete"):
            evaluate_directory(output, metrics=("si_sdr",))
    (output / "metrics.csv").write_text("stale scores", encoding="utf-8")
    write_json(output / "metrics.json", {"stale": True})
    writes.clear()

    def record_save(path, *args, **kwargs):
        writes.append(Path(path).name)
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(torchaudio, "save", record_save)
    run()
    assert writes == ["1.wav"]
    assert file_sha256(output / "0.wav") == first_hash
    assert read_json(output / manifest_name)["status"] == "complete"
    assert not (output / "metrics.json").exists()
    assert not (output / "metrics.csv").exists()
    writes.clear()
    run()
    assert not writes


@pytest.mark.parametrize(
    ("formulation", "steps", "spacing", "t_min", "expected"),
    [
        ("SBVE", 4, "time_uniform", 0.0, [1.0, 0.75, 0.5, 0.25, 0.0]),
        ("OTCFM", 4, "time_uniform", None, [1.0, 1.0 - 0.97 / 3, 1.0 - 0.97 * 2 / 3, 0.03, 0.0]),
        ("OTCFM", 1, "time_uniform", None, [1.0, 0.0]),
        ("OTCFM", 4, "time_uniform", 0.2, [1.0, 0.8, 0.6, 0.4, 0.2]),
        ("OTCFM", 1, "time_quadratic", None, [1.0, 0.03]),
    ],
)
def test_manifest_grid_matches_network_call_times(
    artifact_workspace, monkeypatch, formulation, steps, spacing, t_min, expected
):
    root, _ = artifact_workspace
    model = GenerativeModel4SE(formulation, backbone="artifact_test")
    checkpoint = model.save_checkpoint(root / "grid.safetensors")
    calls = []
    original_forward = ArtifactBackbone.forward

    def record_forward(self, state, time):
        calls.append(float(time[0]))
        return original_forward(self, state, time)

    monkeypatch.setattr(ArtifactBackbone, "forward", record_forward)
    output = run_generative_inference(
        checkpoint=checkpoint,
        num_steps=steps,
        skip_type=spacing,
        t_min=t_min,
        max_samples=1,
        device="cpu",
        progress=False,
    )
    manifest = read_json(output / "inference.json")
    assert manifest["time_grid"] == pytest.approx(expected)
    assert calls == pytest.approx(manifest["time_grid"][:-1], abs=1e-6)
    assert manifest["terminal_time"] == manifest["time_grid"][-1]


def test_checkpoint_evaluation_hash_and_legacy_manifest(artifact_workspace):
    root, _ = artifact_workspace
    model = GenerativeModel4SE("SBVE", backbone="artifact_test")
    checkpoint = model.save_checkpoint(root / "checkpoint.safetensors")
    output = run_generative_inference(checkpoint=checkpoint, device="cpu", progress=False)
    _, metrics_path = evaluate_directory(output, metrics=("si_sdr",), max_workers=1)
    assert read_json(metrics_path)["source_type"] == "checkpoint"
    manifest = read_json(output / "inference.json")
    for key in ("source_type", "model_path", "status"):
        manifest.pop(key)
    write_json(output / "inference.json", manifest)
    evaluate_directory(output, metrics=("si_sdr",), max_workers=1)
    with torch.no_grad():
        model.generator.scale.add_(0.1)
    model.save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="model_sha256"):
        evaluate_directory(output, metrics=("si_sdr",), max_workers=1)


def test_failed_json_write_preserves_previous_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    write_json(path, {"status": "in_progress"})
    with pytest.raises(TypeError):
        write_json(path, {"unserializable": object()})
    assert json.loads(path.read_text()) == {"status": "in_progress"}


@pytest.mark.parametrize("input_kind", ["file", "directory", "output_symlink", "input_symlink"])
def test_overwrite_rejects_input_output_overlap_before_loading(artifact_workspace, input_kind):
    root, dataset = artifact_workspace
    source_dir = dataset / "test/noisy"
    source = source_dir / "0.wav"
    output = source_dir
    input_path = source if input_kind == "file" else source_dir
    if input_kind == "output_symlink":
        output = root / "output_alias"
        output.symlink_to(source_dir, target_is_directory=True)
    elif input_kind == "input_symlink":
        input_path = root / "input_alias.wav"
        input_path.symlink_to(source)
    originals = {p.name: p.read_bytes() for p in source_dir.glob("*.wav")}
    # No model reference: overlap validation must precede model loading or writes.
    with pytest.raises(ValueError, match="input and output paths must not overlap"):
        enhance_audio_files(inputs=[input_path], output_dir=output, overwrite=True)
    assert {p.name: p.read_bytes() for p in source_dir.glob("*.wav")} == originals
    assert not (source_dir / "enhance.json").exists()


@pytest.mark.parametrize("length", [1, 100, 255, 256, 4096])
def test_short_audio_roundtrip_preserves_length_and_normal_audio(length):
    from cof.data.stft import StftTransform, pad_spec

    transform = StftTransform()
    audio = torch.linspace(0.1, 0.3, length).reshape(1, -1)
    spectrum, invert = transform.to_stft(audio)
    restored = invert(spectrum).reshape(-1)
    assert restored.numel() == length
    assert torch.isfinite(restored).all()
    if length < 256:
        torch.testing.assert_close(restored, audio.reshape(-1), atol=1e-6, rtol=1e-5)
    else:
        previous = pad_spec(transform.stft(audio / float(audio.abs().max())).unsqueeze(0))
        torch.testing.assert_close(spectrum, previous, atol=0, rtol=0)
        previous_audio = transform.istft(previous.squeeze(), length=length) * float(audio.abs().max())
        torch.testing.assert_close(restored, previous_audio, atol=0, rtol=0)


def test_empty_audio_has_explicit_error():
    from cof.data.stft import StftTransform

    with pytest.raises(ValueError, match="empty audio"):
        StftTransform().to_stft(torch.empty(1, 0))


def test_short_audio_does_not_abort_enhancement_batch(artifact_workspace):
    root, dataset = artifact_workspace
    short = root / "short.wav"
    sf.write(short, np.linspace(0.1, 0.3, 100), 16000)
    checkpoint = GenerativeModel4SE("SBVE", backbone="artifact_test").save_checkpoint(root / "model.safetensors")
    output, count = enhance_audio_files(
        checkpoint=checkpoint,
        inputs=[short, dataset / "test/noisy/0.wav"],
        output_dir=root / "enhanced",
        device="cpu",
        progress=False,
    )
    assert count == 2
    assert read_json(output / "enhance.json")["status"] == "complete"
    for name, length in (("short.wav", 100), ("0.wav", 4096)):
        audio, sr = sf.read(output / name)
        assert sr == 16000
        assert len(audio) == length
        assert np.isfinite(audio).all()


@pytest.fixture
def metric_cache(tmp_path, monkeypatch):
    from cof.pipeline.evaluation import _calculate_metrics

    scores = {"value": 1.0}
    monkeypatch.setattr(
        "cof.pipeline.evaluation._evaluate_file",
        lambda *args: {"filename": "a.wav", "SI_SDR": scores["value"]},
    )

    def calculate(overwrite=False):
        return _calculate_metrics(
            [(tmp_path / "a.wav", tmp_path / "a.wav")],
            tmp_path,
            {"source_type": "external"},
            sample_rate=16000,
            metrics=("si_sdr",),
            max_workers=1,
            overwrite=overwrite,
        )

    csv, metadata = calculate()
    return calculate, scores, csv, metadata


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("damage", ["truncated", "changed_score", "wrong_summary"])
def test_metric_cache_rejects_corruption(metric_cache, legacy, damage):
    calculate, _, csv, metadata = metric_cache
    record = read_json(metadata)
    if legacy:
        record.pop("csv_sha256")
    write_json(metadata, record)
    calculate()  # Valid legacy caches remain reusable.
    if damage == "truncated":
        csv.write_text("filename,SI_SDR\n")
    elif damage == "changed_score":
        csv.write_text("filename,SI_SDR\na.wav,9.0\n")
    else:
        record["summary"]["SI_SDR"]["mean"] = 99.0
        write_json(metadata, record)
    with pytest.raises(ValueError, match="Invalid metric artifacts"):
        calculate()


def test_failed_csv_staging_keeps_previous_metric_pair(metric_cache, monkeypatch):
    calculate, scores, csv, metadata = metric_cache
    original = (csv.read_bytes(), metadata.read_bytes())
    scores["value"] = 2.0

    def fail_csv(self, path, *args, **kwargs):
        Path(path).write_text("partial")
        raise OSError("simulated CSV failure")

    monkeypatch.setattr("pandas.DataFrame.to_csv", fail_csv)
    with pytest.raises(OSError, match="simulated CSV failure"):
        calculate(overwrite=True)
    assert (csv.read_bytes(), metadata.read_bytes()) == original
    calculate()


@pytest.mark.parametrize("legacy", [False, True])
def test_interrupted_metric_pair_publication_is_detected(metric_cache, monkeypatch, legacy):
    calculate, scores, csv, metadata = metric_cache
    if legacy:
        record = read_json(metadata)
        record.pop("csv_sha256")
        write_json(metadata, record)
    old_json = metadata.read_bytes()
    scores["value"] = 2.0
    original_replace = Path.replace

    def fail_final_json(self, target):
        if Path(target) == metadata:
            raise OSError("simulated publication failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_final_json)
    with pytest.raises(OSError, match="simulated publication failure"):
        calculate(overwrite=True)
    assert metadata.read_bytes() == old_json
    assert "2.0" in csv.read_text()
    with pytest.raises(ValueError, match="Invalid metric artifacts"):
        calculate()
