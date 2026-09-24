# Training: pretraining and CoF post-training

CoF is trained in two stages. The stage is selected explicitly with `--training-stage {pretrain,posttrain}`;
pretraining additionally declares the formulation with `--formulation {SB-VE,OT-CFM}`.

| | Pretraining | Post-training |
| --- | --- | --- |
| Stage flag | `--training-stage pretrain` | `--training-stage posttrain` |
| Formulation | `--formulation SB-VE` or `--formulation OT-CFM` (required) | Inherited from the pretraining run |
| Method | `--pretraining-method` (default `base`) | `--post-training-method` (default `cof`) |
| `training_unit` | `epoch` | `step` |
| Budget | `num_epoch: 1000`, `max_steps: -1` | `max_steps: 4000` |
| Objective | Native-target regression on analytical path states (SB-VE: clean speech, OT-CFM: vector field) | DRC + CTC on self-generated rollout states |
| Initialization | From scratch | `--pretrain-run` (mandatory) |

Both stages add to the same `run_dir`. Each invocation creates `runs/<run-name>/`, where `<run-name>` is
`cof_<formulation>_<stage>_%m%d%H%M` (for example `cof_sbve_pretrain_09241740`) unless `--run-name` says otherwise.

## Inherited defaults

`config/SB-VE/{pre,post}train.yml` and `config/OT-CFM/{pre,post}train.yml` define the released protocol on top
of `config/<formulation>/base.yml`. Pretraining starts from the `pretrain.yml` chain; post-training starts from
the pretraining run's `config.yml` and overlays the stage defaults from the `posttrain.yml` preset. The two
paths differ in one schedule value — `rollout_n_max` is 16 for SB-VE and 8 for OT-CFM:

```yaml
posttrain:
  max_steps: 4000
  validation_every_n_steps: 200
  cof:
    objective:
      si_sdr_weight: 0.01
      magnitude_weight: 0.7
      complex_weight: 0.3
    drc:
      rollout:
        n_max: 16   # SB-VE; 8 for OT-CFM
        fixed_n: null
    ctc:
      lambda: 0.1
      steps: 1
      cf_model: "ema"
      fa_model: "ema"
      shared_noise: true
```

Shared runtime defaults worth knowing before a deployment run: `learning_rate: 1.0e-4`, `batch_size: 8`,
`optimizer: Adam`, `reduction: sum`, `ema: True` with `ema_rate: 0.999`, `seed: 1234`, `logger: wandb`,
`sampling_solver` `SB_SDE_Solver` for SB-VE and `OTCFM_ODE_Solver` for OT-CFM at `num_steps: 4`, `valid_metric_samples: 50`,
`save_state_steps: 1000`, and `checkpoints_total_limit: 3`.

Post-training inherits the following parameters verbatim from the pretraining run and rejects them on the CLI:
`generative_backbone`, `backbone_kwargs`, `formulation_kwargs`, `sampling_num_steps`,
and `sampling_skip_type`. Everything else — optimization, budget, EMA, dataset overrides, and the DRC/CTC
controls below — can be overridden per run. Post-training does not re-read the host `.config/cof.yml`: the
registry, runtime, and optimization settings all come from the pretraining run's `config.yml` snapshot, so
`--dataset` must name an ID already registered in that snapshot.

`--sampling-solver`, `--sampling-num-steps` (default 4), and `--sampling-skip-type` configure the
validation-time inference. The released presets monitor PESQ and SI-SDR through `evaluation.metrics`
(default: `[pesq, si_sdr]`). Offline `cof metric` defaults to all three metrics, including ESTOI.
In post-training the validation sampling solver is by construction the same solver its DRC/CTC rollouts use — both resolve from the single
`formulation.sampling.solver` key: `--sampling-solver` (or a preset) selects it, defaulting to the value
inherited from the pretraining run. The choice is recorded in the run's `config.yml` and acts as the
recommended inference solver — sampling the finished run with any other solver warns. `--sampling-num-steps`
and `--sampling-skip-type` stay locked for post-training, and resuming an interrupted run still requires the
solver it was trained with.

## Step 1 — Pretrain the base model

```bash
uv run cof train \
  --training-stage pretrain \
  --formulation SB-VE \
  --dataset voicebank \
  --yes
```

This stage is epoch-based and trains the selected formulation from scratch. Keep the run name it prints.
The post-training stage needs the exported weights from it.

## Step 2 — Run CoF post-training

Post-training initializes from the pretrained weights and is step-based:

```bash
uv run cof train \
  --training-stage posttrain \
  --pretrain-run runs/cof_sbve_pretrain_MMDDhhmm \
  --dataset voicebank \
  --yes
```

`--pretrain-run` is mandatory: it points at the pretraining run directory whose `config.yml` supplies the
inherited parameters and whose exported weights initialize the new run (resolution order: the recorded default
test model, the best validation PESQ export, `model_last.safetensors`, `model.safetensors`, then the best
validation loss export). The CLI rejects the command before preflight when the flag is missing. The run
configuration records `pretrain_run`, `init_model_path`, and `init_model_sha256`.

> **Release status (peer review):** the CoF post-training implementation is withheld in this snapshot. The
> command above is accepted by the CLI but prints an explanatory notice and exits before any run directory is
> created. Every configuration surface described on this page is already present and becomes effective when
> the implementation returns upon acceptance.

### The CoF objective

CoF post-training applies a single objective — Dynamic Rollout Correction (DRC) plus the Counterfactual
Transition Consistency term (CTC); there is no mode flag. The implementation is withheld during peer review
and is described in the paper (arXiv:2609.24651). The preset defaults are the paper configuration:
`lambda_ctc=0.1`, `ctc_steps=1`, `ctc_cf_model=ema`, `ctc_fa_model=ema`, and shared Brownian noise between the
two branches. Related controls:

- `--lambda-ctc` — weight of the CTC term relative to DRC.
- `--ctc-steps` — number of uniform sub-steps between `t` and `s`; `1` is a single transition.
- `--ctc-cf-model` / `--ctc-fa-model` — `online` or `ema` network for the counterfactual target and the factual
  branch. Choosing `ema` for either requires EMA to be enabled (`--ema`, on by default).
- `--ctc-shared-noise` / `--no-ctc-shared-noise` — whether both branches share one set of Brownian increments.


### Other useful overrides

```bash
--rollout-n-max 16                                 # maximum rollout deployment depth N_max
--rollout-fixed-n 4                                # pin the rollout deployment depth (unset: sampled schedule)
--max-steps / --batch-size / --learning-rate / --seed / --num-workers
--logger none                                      # disable W&B for an unattended host
--comment "..."                                    # free-form note persisted into the run config
```

## Resume an interrupted run

```bash
uv run cof train \
  --training-stage pretrain \
  --dataset voicebank \
  --resume --checkpoint-path runs/cof_sbve_pretrain_MMDDhhmm/checkpoints/last.ckpt \
  --yes
```

`--resume` and `--checkpoint-path` imply each other: passing one without the other is rejected before the run
starts. `--training-stage` is still required and must match the stage of the checkpointed run, and the
component identity — formulation, backbone and kwargs, methods, `--run-name`/`--run-dir` — cannot change while
resuming. Passing the run directory or its `checkpoints/` directory selects `checkpoints/last.ckpt`.
Passing an explicit file such as `checkpoints/step=1000.ckpt` restores that exact state, without falling
back to `last.ckpt`. The file must exist inside the run's `checkpoints/` directory. The run directory is
derived from the checkpoint path, so `--run-name` is not needed. Resuming an existing run does not require
`--pretrain-run`. Intermediate states live in `checkpoints/step=*.ckpt` and are
pruned to `checkpoints_total_limit`; `checkpoints/last.ckpt` is the complete resumable Lightning state.

## Validation, telemetry, and early stopping

In the step-based post-training stage, validation runs every `validation_every_n_steps` steps on
`valid_metric_samples` utterances; the epoch-based pretraining stage validates once per epoch. Each validation
in the post-training stage prints a one-line summary on rank 0 with the step, the validation loss, PESQ, and
SI-SDR. W&B is the default logger (`--logger none` disables it).

### Weights & Biases metric layout

Every run logs its full effective configuration to `wandb.config`. Metrics are namespaced
`{pretrain|posttrain}/{train|valid}/{name}` so the stage groups first in the panel tree:

| Group | Content |
| --- | --- |
| `{stage}/train` | `loss`, `prediction_loss`, the active loss terms (`time_loss` or `magnitude_loss`/`complex_loss`/`si_sdr_loss`); post-training adds `ctc_clean_loss` and the rollout protocol (`rollout_steps`, `corrected_time`, `drc_model_calls`, `ctc_time`, `model_calls`) |
| `{stage}/valid` | the same metric set as `{stage}/train` plus the sampled `PESQ_per_epoch` and `SI_SDR_per_epoch` |

Each scalar value is reported under exactly one name — there are no duplicate aliases.

Early stopping is off by default. Enable it in the inherited preset with
`runtime.early_stopping_enabled: true`. `--patience` configures how many validations without improvement
are tolerated before stopping.

## Run artifacts

A finished run contains:

```text
runs/<run-name>/
  config.yml                              # effective configuration and provenance
  model_valid_loss={loss:.6f}.safetensors # best validation loss (EMA weights)
  model_valid_pesq={pesq:.4f}.safetensors # best validation PESQ (EMA weights); the default inference model
  model_last.safetensors                  # end-of-training EMA weights
  checkpoints/
    step={step}.ckpt                      # resumable states, pruned to checkpoints_total_limit
    last.ckpt                             # latest resumable state
```

Only the best variant of each metric export is kept. All three model exports carry EMA weights; when EMA is
disabled they fall back to the online weights and `weights.last_model_weights` in `config.yml` records the
source. `model_valid_pesq=*.safetensors` is what inference selects by default.

## Preflight and unattended runs

Training preflight probes the host, resolves the effective configuration, and asks for confirmation.
`--yes` skips that confirmation: use it only after reviewing the printed configuration, and only in
non-interactive deployments. The preflight result, including the NCSN++ operator backend
(`host.ncsnpp_operator_backend`, `host.ncsnpp_cuda_jit_status`), is written into the run's `config.yml`.

## GPU memory

Gradient checkpointing is off by default. Enable it only after an actual CUDA OOM. NCSN++ declares
`handles_gradient_checkpointing`, so the pipeline does not wrap the network and `--gradient-checkpointing`
is a no-op. Enable checkpointing inside the backbone instead:

```bash
--backbone-kwargs '{"gradient_checkpointing": true}'
```

The checkpointing granularity is then controlled by
`--backbone-kwargs '{"gradient_checkpointing": true, "gradient_checkpointing_min_level": 0}'`. The default
keeps shallow, high-resolution activations live and checkpoints the deeper levels and middle blocks; `0`
checkpoints every level, trading recomputation for the largest memory saving.

If memory is still short at full checkpointing, reduce `--batch-size`. Setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment helps with fragmentation.

## Component API

Both stage pipelines are Lightning modules. They receive the same standalone
`GenerativeModel4SE` plus a registered method (`base` for pretraining, `cof` for
posttraining). Native losses live in formulation definitions; CoF injects a common
clean-endpoint loss. See [architecture](architecture.md) for extension interfaces.

The preserved YAML presets additionally expose `evaluation.metrics` and
`evaluation.selection_metric`. The latter names one output of an enabled metric;
its metadata determines whether smaller or larger values select the best export.
The default remains PESQ. Run configuration is still written to `config.yml`.

Python model exports use `model.save_checkpoint(path)` and
`GenerativeModel4SE.from_checkpoint(path)`. They contain construction metadata as
well as weights. Use the full Lightning checkpoint for optimizer/EMA continuation.
