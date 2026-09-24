# Datasets, registration, and paired data creation

CoF consumes paired clean/noisy data. A registered dataset is nothing more than an ID and an absolute path in
the host configuration; the training and evaluation commands resolve the ID through that registry.

## The dataset contract

A registered dataset root must contain six directories:

```text
<dataset>/
  train/{clean,noisy}/*.wav
  valid/{clean,noisy}/*.wav
  test/{clean,noisy}/*.wav
```

Registration validates that all six directories exist. Before building the training or validation audio
indices, the dataset validates that the split is non-empty and the clean/noisy WAV filenames match exactly
(`a.wav` is paired with `a.wav`). Inference, metric evaluation, and validation-time sampling use the same
pairing check. A missing counterpart, an extra WAV on one side, or an empty split aborts loading.
Everything downstream identifies an utterance by that shared filename.

Only `*.wav` participates in the contract. `*.flac` and `*.ogg` are accepted as inputs by `dataset create`, but
the exported dataset is WAV.

## Register an existing paired dataset

```bash
uv run cof dataset add --id voicebank --path /path/to/Voicebank+DEMAND --select
uv run cof dataset list
```

`--select` records the ID as the default, so later commands may omit `--dataset`. Registering never copies,
moves, or modifies the source dataset. Other registry commands:

```bash
uv run cof dataset edit --help      # change an ID and/or path
uv run cof dataset delete --help    # remove a registration, leaving files in place
uv run cof dataset inspect --help   # report paired WAV counts in an exported dataset
```

## Create a paired dataset from source corpora

When a paired corpus is not available, `dataset create` synthesizes one by mixing a clean corpus with an
additive-noise corpus and/or a simulated room response.

```bash
uv run cof dataset create \
  --task enhancement \
  --clean wsj0 /path/to/wsj0 \
  --noise wham /path/to/wham \
  --output-dir /path/to/wsj0-wham
```

Supported labels are `--clean {vctk,wsj0,timit}` and `--noise {none,chime,qut,wham}`. Both options take a
label followed by exactly one path. The VCTK and TIMIT partition rules follow StoRM's fixed
train/validation/test logic; the WSJ0 split directories follow RSB (train `si_tr_s`, valid `si_et_05`, test
`si_dt_05`). Either way the split boundaries are reproducible rather than random.

Useful variants:

```bash
# Reverberation only: the --noise none paths must exist but their audio is not mixed.
uv run cof dataset create \
  --task dereverberation \
  --clean wsj0 /path/to/wsj0 \
  --noise none /path/to/wsj0 \
  --output-dir /path/to/wsj0-reverb

# Several tasks in one export.
uv run cof dataset create \
  --task enhancement --task dereverberation \
  --clean wsj0 /path/to/wsj0 \
  --noise wham /path/to/wham \
  --output-dir /path/to/combined
```

`--task` accepts `enhancement` (aliases `enh`, `noise`) and `dereverberation` (alias `derev`); a `+`- or
comma-separated list also works. Synthesis controls and their defaults:

| Option | Default | Meaning |
| --- | --- | --- |
| `--sample-rate` | `16000` | Export sample rate; must match the training contract |
| `--snr-min` / `--snr-max` | `-6.0` / `14.0` | SNR sampling range in dB |
| `--t60-min` / `--t60-max` | `0.4` / `1.0` | Reverberation time range in seconds |
| `--seed` | `100` | Mixing seed |
| `--overwrite` | off | Replace a non-empty output directory |

`--overwrite` recursively deletes the existing output directory before synthesis. Source and output paths
must be separate: equal paths and either direction of directory nesting are rejected, including symbolic
link aliases. Discovered source audio resolving inside the output directory is also rejected before deletion.

The export writes the paired splits plus `create_configuraton.json`, which records the inputs and synthesis
parameters used. Keep it with the dataset: it is the only provenance record of how the pairs were produced.

Then register the export like any other dataset:

```bash
uv run cof dataset add --id wsj0-wham --path /path/to/wsj0-wham --select
```

## Audio and spectrogram contract

`config/dataset.yml` defines 16 kHz mono audio, `n_fft=510`, `hop_length=128`, `num_frames=256`, a
square-root Hann window, and audio normalization. The default spectrogram training path crops or pads each
pair to `(num_frames - 1) * hop_length = 32,640` samples (2.04 seconds). `audio_length: 16000` applies to the
waveform-return path (`return_spec=False`), not the default spectrogram training path. WAV loaders resample
source audio to the configured sample rate when necessary. Inference enhances the full input utterance.
