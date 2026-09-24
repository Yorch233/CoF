# CoF Deployment Agent Guide

CoF is a few-step generative speech enhancement framework that post-trains a pretrained Schrödinger bridge on
the states its own sampler visits. This file tells an execution agent how to install, configure, train, infer,
and evaluate the project on behalf of a human operator. Treat [`README.md`](README.md) as the human overview and
the files below as the command references:

- [`docs/installation.md`](docs/installation.md): host requirements, environment creation, host configuration, and checks
- [`docs/datasets.md`](docs/datasets.md): paired data contract, dataset registration, and dataset synthesis
- [`docs/training.md`](docs/training.md): pretraining, CoF post-training, modes, resume, and run artifacts
- [`docs/inference.md`](docs/inference.md): sampling protocols and result variants
- [`docs/metrics.md`](docs/metrics.md): metric names, run-linked and third-party evaluation

Preserve the `For Human`, `For Agent`, and `Skip This README` hand-off prompts in `README.md`.

## Deployment objective

The standard deployment produces an auditable sequence of artifacts:

```text
paired dataset
  → stage-1 pretraining run
  → CoF post-training run
  → manifested few-step inference WAVs
  → per-file and aggregate metrics
```

Provenance is enforced by the tooling rather than by convention: a run records the configuration and the
initialization weights it used, an inference result records the exact model hash and sampling protocol, and
metric evaluation refuses to score a result whose manifest disagrees with the run it is attributed to. Do not
copy WAV files or checkpoints between run and result directories by hand; a directory that contains WAVs but no
manifest is rejected, and that rejection is a feature. Use the CLI so every artifact keeps its provenance:
`cof run export` is the sanctioned way to copy a run's resolved model elsewhere, and `cof enhance` enhances
loose audio files outside the paired-dataset contract.

## Host and environment rules

- Run every command from the repository root, as `uv run cof ...`.
- Python 3.12 and UV. Create the environment with `uv sync`.
- Training and CUDA sampling require a CUDA-capable host. PESQ, ESTOI, and SI-SDR can run on CPU.
- NCSN++ custom CUDA operators are JIT-compiled on first use. A failed compilation silently falls back to the
  portable native PyTorch operators, which is slower but correct; the outcome is recorded in the run
  configuration as `ncsnpp_operator_backend` and `ncsnpp_cuda_jit_status`. When a deployment must not silently
  lose the optimized path, pass `--require-cuda-jit` so the run aborts instead. Report which backend was used.
- Training preflight prints the effective configuration and asks for confirmation. Use `--yes` only when the
  human has reviewed that configuration, or when the job is genuinely unattended.
- Never write credentials, remote host details, machine paths, dataset paths, checkpoint paths, generated audio,
  or tokens into tracked files. Host-specific settings belong in the ignored `.config/cof.yml`; the dataset
  registry is host-local by design.

## Dataset contract

A registered dataset root must contain:

```text
train/{clean,noisy}/*.wav
valid/{clean,noisy}/*.wav
test/{clean,noisy}/*.wav
```

Clean and noisy filenames must match exactly within a split, and no split may be empty. An utterance is
identified by that shared filename everywhere downstream. Registration stores only the ID and path and must
never copy, modify, or delete the source dataset.

`cof dataset create` can build pairs from separate corpora with `--task enhancement` and/or
`--task dereverberation`, `--clean {vctk,wsj0,timit}`, and `--noise {none,chime,qut,wham}`. It writes the
splits plus `create_configuraton.json`, which records the inputs and synthesis parameters; treat that file as
the dataset's provenance record and keep it. Note the filename spelling — it is what the tool writes.

## Configure the deployment

```bash
uv sync
uv run cof dataset add --id voicebank --path /path/to/dataset --select
uv run cof config
uv run cof dataset list
```

`cof config` writes host settings to `.config/cof.yml`: dataset registry and selection, precision, multi-GPU
and device selection, logger, logging and resumable-state intervals, intermediate-checkpoint limit, run and
result roots. The legacy `inherit` field is not consumed by the training CLI. The stage and formulation
flags select the tracked preset. Confirm that `cof dataset list` shows the intended selected dataset before
training.

Tracked configuration inheritance is:

```text
config/dataset.yml
  └─ config/<formulation>/base.yml   e.g. config/SB-VE/base.yml
       └─ config/<formulation>/{pretrain,posttrain}.yml
            └─ CLI flags             per-item overrides
```

`.config/cof.yml` holds host settings only (dataset registry and selection, precision, multi-GPU and device
selection, logger, logging and checkpoint intervals, run and result roots); the training CLI no longer follows
its `inherit` field. Consequences an agent must respect:

- `--training-stage {pretrain,posttrain}` selects the stage explicitly. Pretraining additionally requires
  `--formulation {SB-VE,OT-CFM}`, which selects `config/<formulation>/pretrain.yml`.
- Post-training requires `--pretrain-run <pretraining-run>`; `--post-training-method` defaults to `cof`. It reads that
  run's `config.yml` and inherits the model/protocol identity parameters verbatim (formulation, backbone and
  kwargs, formulation kwargs, training target, sampling steps and schedule); passing them on the CLI — e.g.
  `--generative-backbone` or `--sampling-num-steps` — is rejected. `--sampling-solver`/`--sampling-num-steps`
  (default 4)/`--sampling-skip-type` configure validation-time inference; the solver is the exception to the
  inheritance: post-training selects it with `--sampling-solver` (default: inherited), it is recorded in the
  run config as the recommended inference solver, and inferring with any other solver warns.
- Per-run overrides such as `--max-steps`, `--batch-size`, and `--dataset` change that run only.

## Execute the training sequence

Pretrain the base model (epoch-based):

```bash
uv run cof train --training-stage pretrain --formulation SB-VE --dataset voicebank --yes
```

Record the exact `cof_<formulation>_<stage>_<MMDDhhmm>` name the command prints; do not attempt to guess or "select the
latest" run later. Then run CoF post-training (step-based):

```bash
uv run cof train \
  --training-stage posttrain \
  --pretrain-run runs/cof_sbve_pretrain_MMDDhhmm \
  --dataset voicebank \
  --yes
```

`--pretrain-run` is mandatory for every post-training run and is never resolved automatically: the stage reads
that run's `config.yml` for the inherited parameters and picks the initialization weights from its exports
(recorded default test model, best validation PESQ, `model_last.safetensors`, `model.safetensors`, best
validation loss). The run configuration records `pretrain_run`, `init_model_path`, and `init_model_sha256`.

Release status: in this snapshot the CoF post-training implementation (DRC + CTC) is withheld during peer
review. The command above prints an explanatory notice and exits before creating a run directory; do not
report a post-training deployment as complete in this state. The configuration surface, inference protocol,
and all weights remain fully available, and the implementation returns upon acceptance.

CoF post-training applies a single objective, `drc_ctc`: Dynamic Rollout Correction on self-generated rollout
states plus the Counterfactual Transition Consistency term; there is no `--post-training-mode` flag. With the
preset defaults (`lambda_ctc=0.1`, `ctc_steps=1`, `ctc_cf_model=ema`, `ctc_fa_model=ema`, shared noise) it is
the configuration reported in the paper. Choosing `ema` for a CTC branch requires EMA to be enabled.

To resume, pass `--resume` together with `--checkpoint-path`; each requires the other, and the run directory is
derived from the checkpoint path. A run or `checkpoints/` directory selects `last.ckpt`; an explicit
`checkpoints/*.ckpt` file restores that exact state. Missing states are rejected without fallback.
Resume does not need `--pretrain-run`.

## Execute inference and metrics

```bash
uv run cof inference \
  --run cof_sbve_posttrain_MMDDhhmm --dataset voicebank --split test \
  --num-steps 4

uv run cof metric --dir results/cof_sbve_posttrain_MMDDhhmm/SB_SDE_Solver_N=4 --metrics pesq,estoi,si_sdr
```

- Exactly one of `--run` or `--ckpt` is required. `--run` resolves a local run, while `--ckpt` accepts a
  local self-describing safetensors model without a run directory. Both produce manifested results that
  `cof metric --dir` can evaluate, with model SHA-256 verification. CoF never downloads a default checkpoint,
  so do not tell a human that a released model will be fetched automatically.
- The model is taken from the run in this order: the recorded default test model, the best validation PESQ
  export, `model_last.safetensors`, `model.safetensors`, then the best validation loss export. Report which one
  was actually used when it matters.
- Each sampling protocol writes its own variant directory (`<solver>_N=<steps>`, e.g. `SB_SDE_Solver_N=4`, `OTCFM_ODE_Solver_N=1`). NFE is part of the identity of a result, so never compare
  or merge numbers across variants.
- Re-running a variant whose recorded signature conflicts raises an error instead of mixing outputs. Treat that
  as a provenance guard: investigate before passing `--overwrite`.

Metric evaluation accepts exactly one input mode: `--dir`, `--run [--result]`, or
`--clean/--noisy/--enhanced` (all three together, with identical non-empty WAV filename sets). The supported
metric names are exactly `pesq`, `estoi`, and `si_sdr` — there is no `dnsmos`, `utmos`, or `nisqa_v2` metric
in this release, and an unsupported name is rejected rather than ignored. Omitting `--metrics` computes the
three intrusive core metrics; all of them are calculated through TorchMetrics:

```bash
uv run cof metric --dir results/cof_sbve_posttrain_MMDDhhmm/SB_SDE_Solver_N=1 \
  --metrics pesq,estoi,si_sdr
```

## Artifact acceptance checks

A completed training run must contain:

```text
runs/<run-name>/
  config.yml
  model_valid_pesq={pesq}.safetensors     # or model_valid_loss=... / model_last.safetensors
  checkpoints/last.ckpt
```

`model_valid_pesq=*.safetensors` is the validation-selected inference default, `model_valid_loss=*.safetensors`
is the diagnostic export, `model_last.safetensors` holds the end-of-training EMA weights, and
`checkpoints/last.ckpt` is the complete resumable Lightning state. Exports are EMA weights; when EMA is
disabled they fall back to the online weights and `last_model_weights` in `config.yml` records the source.

A completed inference result must contain `inference.json` with `status: complete` and exactly the expected
WAV filename set. The manifest records the model source and actual sampling time grid. An interrupted
`in_progress` result must be resumed before evaluation. A completed evaluation must contain both `metrics.csv` and `metrics.json` in the evaluated directory, with a
`summary` block holding the per-column mean and standard deviation.

Do not report a deployment as complete when a required manifest, checkpoint, WAV, or metric file is missing.
Consult the troubleshooting section of the corresponding guide before changing command parameters or
overwriting artifacts.

## GPU memory

Gradient checkpointing is off by default; enable it only after an actual CUDA OOM, and widen it in order:

1. `ncsnpp` checkpoints internally, so `--gradient-checkpointing` is a no-op. Enable it with
   `--backbone-kwargs '{"gradient_checkpointing": true}'`.
2. Raise the granularity to every level with
   `--backbone-kwargs '{"gradient_checkpointing": true, "gradient_checkpointing_min_level": 0}'`.
3. If memory is still short, reduce `--batch-size`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps
   with fragmentation.

## Deployment reporting

When handing a result back to the human, state:

1. the registered dataset ID and the split used;
2. the exact pretraining and post-training run names, plus the `--pretrain-run` used by post-training;
3. whether CUDA JIT or the portable PyTorch operators were used, and whether `--require-cuda-jit` was set;
4. the sampler, step count (NFE), skip type, and seed of each inference variant;
5. the run, result, checkpoint, manifest, and metric artifact paths;
6. which model export inference resolved;
7. any skipped stage, fallback, warning, or incomplete artifact.
