# Metrics

Metric calculation takes exactly one of three input modes and writes a per-file CSV plus a summarized JSON.

| Mode | Selection | Use |
| --- | --- | --- |
| Manifested result | `--dir results/<run>/<variant>` | Evaluate a result produced by `cof inference` |
| Run-linked result | `--run <run> [--result <variant>]` | Resolve the variant directory from a run |
| Third-party directories | `--clean/--noisy/--enhanced` | Evaluate WAV directories produced elsewhere |

Exactly one mode must be given. `--result` requires `--run`, and the three third-party paths must be passed
together.

## Metric names

The supported metric names are `pesq`, `estoi`, and `si_sdr`. Names are accepted
comma-separated and repeatable, are normalized to lower case with `-` mapped to `_`, and are validated: an
unsupported name is rejected with the valid choices listed rather than silently ignored.

| Name | Type | Reference | Reported column(s) |
| --- | --- | --- | --- |
| `pesq` | Intrusive | clean | `PESQ` |
| `estoi` | Intrusive | clean | `ESTOI` |
| `si_sdr` | Intrusive | clean | `SI_SDR` |

All three metrics are computed through TorchMetrics (`PerceptualEvaluationSpeechQuality`,
`ShortTimeObjectiveIntelligibility` in extended mode, and `ScaleInvariantSignalDistortionRatio`). The `pesq`
package remains a required dependency as the PESQ backend. `pystoi` is the backend used internally by
TorchMetrics for STOI/ESTOI, and is also used by dataset synthesis for quality diagnostics. Omitting
`--metrics` computes exactly these three metrics.

```bash
# Core protocol (the default metric set).
uv run cof metric --dir results/cof_sbve_posttrain_MMDDhhmm/SB_SDE_Solver_N=4

# Explicitly the same three metrics.
uv run cof metric --dir results/cof_sbve_posttrain_MMDDhhmm/SB_SDE_Solver_N=4 \
  --metrics pesq,estoi,si_sdr
```

Calculation is concurrent by default: `--max-workers 0` uses the executor's default worker count, and
`--max-workers 1` evaluates the utterances sequentially. PESQ only supports 8 kHz and 16 kHz input, and the
whole protocol is defined at 16 kHz.

## Run-linked evaluation

```bash
uv run cof metric --run cof_sbve_posttrain_MMDDhhmm --result SB_SDE_Solver_N=4
```

A run has one result directory per sampling variant. When a run holds exactly one manifested variant, the
command resolves it directly; with several variants it opens an interactive picker in a terminal and demands
`--result` in a non-interactive shell. Only directories containing `inference.json` are offered as candidates.

Run-linked evaluation is deliberately strict about provenance. Before scoring a single file it verifies that
the manifest records the same dataset ID, split, run name, and model SHA-256 as the run being evaluated. A
mismatch aborts the command, so metrics can never be attributed to the wrong weights, and a result directory
without a manifest cannot be scored at all. List the expected files from the manifest and check that the
enhanced WAVs match it. Unknown names abort the run. `--dir` evaluation goes through the same checks.
It resolves the model according to the manifest's `source_type`: a local run for `run`, or the self-describing `model_path` for `checkpoint`. Checkpoint-based
results therefore require no run directory. Both paths verify the model SHA-256 before scoring or reusing
metrics. Legacy manifests are still readable, including bare-checkpoint manifests that used `run_path`.
Results with `status: in_progress` must finish inference before evaluation.

## Third-party evaluation

```bash
uv run cof metric \
  --clean /path/to/clean --noisy /path/to/noisy --enhanced /path/to/enhanced \
  --sample-rate 16000 \
  --metrics pesq,estoi,si_sdr
```

All three directories must contain the same non-empty set of `*.wav` filenames. `metrics.csv` and
`metrics.json` are written into the enhanced directory, and the summary records the three directories as
`source_type: external`.

## Output artifacts

Both modes write the same pair of files into the evaluated directory:

- `metrics.csv` — one row per utterance, with the `filename` column and every computed score column.
- `metrics.json` — `artifact_type: metrics`, the provenance metadata, the sample rate, the selected metric
  names, `num_files`, and a `summary` mapping each column to its `mean` and `std`.

`summary` skips the `filename` column and aggregates every score column. Existing metrics are not recomputed
when both artifacts match the request and pass integrity checks. Reuse validates the filename set,
columns, row count, finite scores, summary statistics, and the CSV hash when present. New outputs are
staged before atomic replacement of each file, with `csv_sha256` in the JSON binding it to the CSV.
A crash between the two replacements is detected on reuse. Legacy outputs without a hash still undergo
the structural and summary checks. Partial, inconsistent, or conflicting artifacts require `--overwrite`.

## Adding a metric

The three metrics above are the built-ins. Additional metrics can be registered
with `MetricRegister` and selected without changing validation or offline pipeline
code. The component declares its output names, reference requirement and selection
direction. Both callers use `MetricSuite`; see [architecture](architecture.md).
