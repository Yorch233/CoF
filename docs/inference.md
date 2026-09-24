# Inference

Inference enhances one registered dataset split with a trained run and writes a manifested result directory.
It never writes loose WAV files: every result directory carries an `inference.json` recording which model,
dataset, split, and sampling configuration produced it.

## Enhance a split

```bash
uv run cof inference \
  --run cof_sbve_posttrain_MMDDhhmm \
  --dataset voicebank \
  --split test \
  --num-steps 4
```

`--run` names a local run directory (or the run name below `run_dir`); alternatively `--ckpt` points
directly at a self-describing model checkpoint file with no run directory at all. With `--ckpt` the
construction and sampling defaults come from the metadata embedded in the file, the result directory is
named after the file stem, and the dataset still resolves through the host registry. The two flags are
mutually exclusive and one of them is required. CoF does not
download a default checkpoint, so a deployment must supply a local run or checkpoint. When `--run` is given,
the model is resolved from the run directory in this order, first existing file wins:

1. `default_test_model` recorded in the run configuration — the best validation PESQ export,
2. `model_valid_pesq=*.safetensors` with the highest score,
3. `model_last.safetensors`,
4. `model.safetensors`,
5. `best_valid_loss` / `model_valid_loss=*.safetensors` as a diagnostic fallback.

If none exists, the command fails rather than guessing.

## Sampling controls

| Option | Default | Meaning |
| --- | --- | --- |
| `--sampler` | from the run config | `SB_SDE_Solver`, `SB_ODE_Solver`, `OTCFM_ODE_Solver`, or `AUTO`; overrides the stored solver |
| `--num-steps` | from the run config | Number of function evaluations (NFE) |
| `--skip-type` | from the run config | Time-grid spacing: `time_uniform` or `time_quadratic` |
| `--split` | `test` | `train`, `valid`, or `test` |
| `--max-samples` | all pairs | Evaluate only a random subset of the split |
| `--seed` | `1234` | Seed for subset selection and sampler noise |
| `--device` | `auto` | Torch device or CUDA index |
| `--num-workers` | `0` | Data-loader worker processes; zero loads inline |
| `--t-min` | formulation-dependent | Formulation-dependent lower time bound. SB-VE defaults to terminal time 0. OT-CFM's default uniform grid uses safe network-call times and then advances to 0 (see below) |
| `--overwrite` | off | Reset the variant directory and re-infer from scratch |

The released defaults sample SB-VE with `SB_SDE_Solver` and OT-CFM with `OTCFM_ODE_Solver` at 4 steps; both are the formulations' canonical solvers, so the flag is only needed to select a non-default one such as SB-VE's probability-flow `SB_ODE_Solver`. Few-step evaluation is the point
of the method: report the exact step count with every result, because results at different NFE are not
comparable.

For OT-CFM's default uniform grid, the network-call times are `linspace(1, 0.03, N)` followed by a
transition to zero. At `N=1` the single call is at 1 and the transition goes directly to zero. A different
positive `--t-min` truncates integration at that value. With `time_quadratic`, the configured positive
lower bound is the terminal time. Each manifest records the actual `time_grid` (all `N+1` endpoints) and
its final value as `terminal_time`. The first `N` points are the network evaluation times.

## Result directory

The result directory is `results/<run-name>/<variant>/`, where the variant encodes the sampling protocol:

```text
results/cof_sbve_posttrain_MMDDhhmm/
  SB_SDE_Solver_N=4/
    inference.json            # provenance manifest
    <utterance>.wav           # one file per paired test utterance
```

The variant name is built from the solver and step count only — `<solver>_N=<steps>`, for example
`SB_SDE_Solver_N=4` or `OTCFM_ODE_Solver_N=1`. The remaining protocol details (skip type, lower time bound, subset size, seed) are not part of
the directory name; they live in the manifest signature, so re-running the same variant under a different
protocol or seed raises a signature conflict instead of silently mixing outputs.

`inference.json` records `source_type` (`run` or `checkpoint`), the absolute `model_path`, the dataset ID
and split, the run name and path, the run ID, the model SHA-256, the sample rate and device, the sampler,
step count, skip type, lower time bound, actual `time_grid`, `terminal_time`, subset size, and seed, together
with the expected output `files` and a `status` of `in_progress` or `complete`. Its `model_sha256` makes an
inference result traceable to exact weights.

Before writing audio, inference atomically publishes an `in_progress` manifest with the requested
signature and complete expected filename list. Every WAV is written privately and atomically renamed into
place only after encoding succeeds. After all expected outputs exist, the manifest is atomically marked
`complete`. Re-running a variant is resumable by default: when the manifest matches the requested signature
and filename list, completed WAVs are skipped and only missing files are inferred. This also recovers a
first run that was interrupted. Evaluation rejects an `in_progress` result until inference completes it.
If missing audio is regenerated, existing metric artifacts are invalidated. Older manifests without the new
signature fields require an explicit `--overwrite` to regenerate with the complete protocol record.
An existing manifest whose recorded signature disagrees with the new request raises an error instead of silently mixing outputs, and a result
directory that contains WAV files but no manifest is rejected outright. Use `--overwrite` to deliberately
replace a result: it resets the whole variant directory — WAVs, `inference.json`, and any `metrics.*` files —
and infers everything from scratch.

OT-CFM only accepts `OTCFM_ODE_Solver`; other solvers are rejected for an OT-CFM run. SB-VE additionally
offers the probability-flow `SB_ODE_Solver`. A post-trained run records the solver its rollouts were
generated with; sampling that run with a different solver warns that the result departs from the
post-trained configuration.

## Batch pattern

For the full few-step protocol, run one inference per step count. Each command writes its own variant
directory, so nothing is overwritten:

```bash
for n in 1 4 16; do
  uv run cof inference \
    --run cof_sbve_posttrain_MMDDhhmm --dataset voicebank --split test \
    --num-steps "$n"
done
```

Progress is shown by default and can be silenced with `--no-progress` for log-based deployments.

## Enhance individual files

Outside the paired-dataset contract — demos, field recordings, deployment smoke tests — `cof enhance`
enhances loose audio files with the same run-resolved model and the same sampling controls:

```bash
uv run cof enhance \
  --run cof_sbve_posttrain_MMDDhhmm \
  --input noisy.wav \
  --output-dir enhanced/
```

`--input` accepts a file or a directory of audio files and repeats. The output directory receives one
enhanced WAV per input plus an `enhance.json` manifest recording the run, model SHA-256, sampling protocol,
and seed, with the same conflict and resume semantics as inference variants. With `--overwrite`, input
and output paths must not overlap, including paths resolved through symbolic links.

Non-empty audio shorter than the STFT reflection-padding minimum is zero-padded on the right to one FFT window before
transformation, and the enhanced output is trimmed to the original sample count. With the default
`n_fft=510`, only inputs shorter than 256 samples take this path. Empty audio is rejected with an explicit
error. Longer inputs keep the existing processing.

When corresponding clean and noisy references are available, score enhanced output through
`cof metric --clean CLEAN --noisy NOISY --enhanced ENHANCED`. All three directories must contain the same
non-empty WAV filename set. `--dir` requires an `inference.json` and does not accept `enhance.json`.

## Export a model

`cof run export <run> --output-dir DIR` writes three files into `DIR`: the model inference would resolve —
the same priority order — renamed to `<dataset>_<formulation>[_cof].safetensors` (for example
`voicebank_sbve_cof.safetensors`; the `_cof` suffix marks the CoF stage), the run's `config.yml` with
`weights.default_test_model` rewritten to match, and an `export.json` provenance record holding the run
identity, the original and exported model filenames, the model SHA-256, and the CoF version. The export keeps
a complete `config.yml` + model pair, so it can itself be passed to `--run` later. It is the sanctioned way
to export a run with its provenance record. Copying a self-describing checkpoint preserves its embedded
construction metadata and weights, but does not create the additional `export.json` record.
