"""Dataset synthesis must never delete its source corpora."""

import importlib

import pytest

creation = importlib.import_module("cof.data.create_dataset")


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("relation", ["same", "parent", "child", "symlink"])
@pytest.mark.parametrize("input_kind", ["clean", "noise"])
def test_dataset_rejects_overlapping_paths(tmp_path, overwrite, relation, input_kind):
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "original.wav"
    marker.write_bytes(b"original source")
    other = tmp_path / "other"
    other.mkdir()
    if relation == "same":
        output = source
    elif relation == "parent":
        output = tmp_path
    elif relation == "child":
        output = source / "export"
    else:
        output = tmp_path / "alias"
        output.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="must not overlap"):
        creation.create_dataset(
            tasks=["enhancement"],
            clean_dataset=creation.CleanDataset.WSJ0,
            clean_inputs=[source if input_kind == "clean" else other],
            noise_dataset=creation.NoiseDataset.WHAM,
            noise_inputs=[source if input_kind == "noise" else other],
            output_dir=output,
            overwrite=overwrite,
        )
    assert marker.read_bytes() == b"original source"


def test_dataset_rejects_discovered_source_in_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "original.wav"
    marker.write_bytes(b"original source")
    alias = source / "alias.wav"
    alias.symlink_to(marker)
    monkeypatch.setattr(creation, "discover_clean_dataset_splits", lambda *args: {"train": [alias]})
    monkeypatch.setattr(creation, "discover_noise_pools", lambda *args: ({}, []))
    with pytest.raises(ValueError, match="must not overlap"):
        creation.create_dataset(
            tasks=["dereverberation"],
            clean_dataset=creation.CleanDataset.WSJ0,
            clean_inputs=[source],
            noise_dataset=creation.NoiseDataset.NONE,
            noise_inputs=[source],
            output_dir=output,
            overwrite=True,
        )
    assert marker.read_bytes() == b"original source"
