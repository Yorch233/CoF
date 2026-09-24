# Installation and runtime configuration

Clone the repository and run every command from its root:

```bash
git clone https://github.com/Yorch233/CoF.git
cd CoF
```

All examples use `uv run cof ...` so that the project virtual environment is used consistently instead of
a shell-activated interpreter.

## Host requirements

| Item | Requirement |
| --- | --- |
| Operating system | Linux |
| Python | 3.12 (`.python-version`; `requires-python = ">=3.12,<3.13"`) |
| Package manager | [UV](https://docs.astral.sh/uv/) |
| CUDA toolkit | Needed on hosts where the NCSN++ JIT operators compile (`nvcc`) |
| GPU | Required for training and for CUDA sampling |

All three supported metrics — PESQ, ESTOI, and SI-SDR — run comfortably on a CPU-only host. Training and
CUDA sampling expect a CUDA-capable deployment host.

## Step 1 — Set up the environment

```bash
uv sync
```

`uv sync` creates `.venv/` and installs the dependency set from `pyproject.toml`. One dependency detail
matters for a deployment host:

- Torch and Torchaudio are pinned to `>=2.7,<3`. Install a build that matches the host CUDA driver before
  expecting the NCSN++ CUDA operators to compile.

Verify the environment:

```bash
uv run python --version
uv run cof --help
```

## Step 2 — Select the generative backbone

`config/SB-VE/base.yml` and `config/OT-CFM/base.yml` default to `model.backbone: ncsnpp_base` (selected on
the CLI through `--generative-backbone`) and take their options through `--backbone-kwargs` (a JSON object).
The backbone is a pretraining-time choice: post-training inherits it from the pretraining run.

## Step 3 — Configure the host

```bash
uv run cof config
```

The wizard writes host-specific settings to the ignored `.config/cof.yml`:

| Field | Meaning |
| --- | --- |
| `datasets` | Dataset registry: `id -> absolute path` |
| `dataset` | Registry ID used when a command omits `--dataset` |
| `mixed_precision` | Trainer precision: `none`, `fp16`, or `bf16` |
| `multi_gpu`, `gpu_ids` | Device selection; `gpu_ids` may be a list |
| `logger` | `wandb` or `none` |
| `log_steps`, `save_state_steps` | Logging interval and resumable-state interval |
| `checkpoints_total_limit` | How many intermediate checkpoints to retain |
| `run_dir`, `results_dir` | Roots for training runs and inference results |
| `inherit` | Legacy preset selection; not consumed by the current training CLI |

`run_dir` and `results_dir` default to `runs/` and `results/` inside the project. Both are ignored by Git.

## Configuration inheritance

Tracked presets are layered, and each layer overrides the one above it:

```text
config/dataset.yml              spectrogram and audio contract (sample rate, FFT, hop, frame count)
  └─ config/SB-VE/base.yml      model, optimization, sampling, and runtime defaults
       └─ config/SB-VE/pretrain.yml | config/SB-VE/posttrain.yml    stage defaults
            └─ CLI flags             per-item overrides
```

The stage and formulation are selected per invocation rather than through `inherit`:

- `--training-stage pretrain --formulation SB-VE|OT-CFM` starts from `config/<formulation>/pretrain.yml` and
  overlays the host settings from `.config/cof.yml`.
- `--training-stage posttrain --pretrain-run <run>` (`--post-training-method` defaults to `cof`) starts from the pretraining
  run's persisted `config.yml`: model- and protocol-defining parameters (formulation, backbone and kwargs,
  formulation kwargs, training target, sampling step count and skip type) are inherited verbatim and cannot
  be overridden on the CLI. The solver is an exception: `--sampling-solver` selects the rollout and validation
  solver, defaulting to the inherited solver. Stage defaults come from `config/<formulation>/posttrain.yml`.
- `config/default.yml` remains as a convenience preset but is no longer consumed by the training CLI.

## CUDA JIT operators

NCSN++ carries custom CUDA operators that compile on first use. Training preflight probes them and writes the
outcome into the run configuration:

- `host.ncsnpp_operator_backend` — the operator implementation that will actually be used.
- `host.ncsnpp_cuda_jit_status` — the JIT compilation result.

A failed compilation prints the error and, in interactive mode, asks for confirmation before continuing with
the portable native PyTorch operators, which are correct but slower; with `--yes` the confirmation is skipped
and the fallback applies directly. When a deployment must not silently lose the optimized path, pass
`--require-cuda-jit`; the run then aborts instead of falling back.

## Checks

```bash
uv run pytest -q
uv run ruff format --check cof test
uv run ruff check cof test
uv run cof run list --help
```

`cof run list` prints the runs below the configured run root, `cof run show <run>` summarizes one run's
persisted configuration and exported weights, and `cof run results <run>` lists its manifested inference
variants with their metric summaries — the operator's read-only view of a deployment. The same
guide's tests double as a deployment self-check: the end-to-end suite drives the full CLI on a synthetic
dataset and needs no GPU or external data.
