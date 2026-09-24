# Component architecture

The core model is an ordinary `torch.nn.Module`. Mathematical formulations,
training algorithms, Lightning execution and file provenance have separate owners.
This refactor removes the old Python entry points; checkpoint loading retains only
a limited compatibility path for legacy exports (see "Configuration and checkpoints").

## Source layout

```text
cof/
  model.py                       GenerativeModel4SE
  backbone/                      named neural architectures
  formulation/
    base.py                      Dynamics, SDE, ODE, Formulation, Sampler
    registry.py                  formulation registration and solver metadata
    sb_ve/
      definition.py              SBVESDE, SBVEFormulation, native pretraining loss
      sampling.py                SBVESampler base; SB_SDE_Solver, SB_ODE_Solver
    ot_cfm/
      definition.py              OTCFMODE, OTCFMFormulation, native pretraining loss
      sampling.py                OTCFM_ODE_Solver
  method/
    base.py                      TrainingMethod, StepContext, StepResult
    registry.py                  independent pretraining/posttraining registries
    pretrain/base.py             BasePretraining
    posttrain/cof/
      method.py                  CoFPosttraining composition
      drc.py                     dynamic schedule and corrected-state construction
      ctc.py                     matched factual/counterfactual transitions
      loss.py                    CoFCleanLoss
  pipeline/
    build.py                     configuration-to-component construction
    base.py                      common Lightning optimization and logging
    pretrain.py                  PretrainPipeline (LightningModule)
    posttrain.py                 PosttrainPipeline (LightningModule)
    inference.py                 waveform inference and manifested outputs
    evaluation.py                offline evaluation and result tables
    provenance.py                run/dataset resolution, hashes and manifests
  training/
    runner.py                    run creation, data, callbacks and Trainer.fit
    runtime.py                   accelerator/distributed/runtime resolution
    preflight.py                 backbone runtime checks
    callbacks/                   EMA, sampled validation, export and checkpoints
  evaluation/
    base.py                      audio metric metadata and scoring support
    registry.py                  metric registration
    intrusive.py                 PESQ, ESTOI and SI-SDR implementations
    suite.py                     shared MetricSuite
  config/
    manager.py                   existing Config and YAML inheritance
    resolver.py                  preset/host/CLI merge and stage validation
    schemas.py                   validated component settings
  data/                          paired data and per-instance StftTransform
  cli/                           commands and user interaction
```

There is no separate `contracts`, `assembly`, `engine`, `artifacts` or `workflows`
package. Contracts live next to their owners in `base.py`; `pipeline/build.py`
constructs the components; `training/` owns Lightning machinery; provenance belongs
to `pipeline/`. Inference and evaluation remain ordinary functions rather than
Lightning modules because they do not need optimization hooks.

## Model and mathematics

`GenerativeModel4SE(formulation=..., backbone=...)` constructs the named components.
It registers the backbone as `generator` and the formulation as `formulation`.
The formulation owns its dynamics exactly once. The model owns a `StftTransform`
instance, so constructing a dataset or another model cannot alter its representation.

- `forward(state, time, condition)` returns the native prediction: clean data for
  SB-VE, a vector field for OT-CFM.
- `predict_clean(state, time, condition)` converts the same native prediction to
  the common clean domain without detaching gradients.
- `sample(...)` creates an independent sampler, runs it without gradients and
  returns the final spectrum plus state/native-prediction traces. It restores the
  caller's training mode and reports consumed network evaluations.
- `sampling_time_grid(...)` resolves the actual integration endpoints without network evaluation. Inference
  records those points in its manifest and passes the same grid to sampling.
- `enhance(audio, ...)` owns waveform normalization, STFT, sampling and inversion.
- `save_checkpoint`, `load_checkpoint`, and `from_checkpoint` store the
  constructor metadata and weights in a self-describing safetensors file.

`SDE` and `ODE` are explicit mathematical bases in `formulation/base.py`.
`SBVESDE` defines the VE coefficients, conditional marginal, bridge score and
stochastic/probability-flow transitions. `OTCFMODE` defines the conditional Gaussian
path, vector-field target, clean conversion and deterministic Euler update.
The concrete definitions are in their formulation's `definition.py`.

The formulation is the boundary consumed by methods: analytic state sampling,
native target construction, clean conversion, native and clean-driven transitions,
loss calculation and sampler construction. A method never branches on `SBVE` or
`OTCFM`. Formulation metadata owns solver support, constructor defaults, safe
training times and inference lower-time semantics. Configuration resolves those
properties through registered components rather than a second mathematical table.

The SB-VE native loss retains weighted complex regression and waveform L1.
OT-CFM retains native vector-field regression. Both implementations are directly
inside the respective `definition.py`. The CoF endpoint discrepancy lives in
`method/posttrain/cof/loss.py`; its implementation is withheld during peer
review and will be released upon acceptance, while the class and configuration
surface remain as documented.

## Calls during training and inference

```text
pretrain pipeline.training_step
  -> BasePretraining.step
     -> formulation.sample_training_state(clean, noisy, time)
     -> model.forward(state, time, [noisy])
     -> formulation.training_loss(native_prediction, ...)
  -> StepResult -> Lightning backward/optimizer

posttrain pipeline.training_step
  -> CoFPosttraining.step
     -> compute_drc -> native model prediction -> formulation.to_clean_prediction
     -> compute_ctc -> formulation.transition_from_clean -> model.predict_clean
     -> shared injected clean-endpoint loss
  -> StepResult -> Lightning backward/optimizer

inference.run_generative_inference
  -> provenance.resolve_model_reference -> GenerativeModel4SE.from_checkpoint
  -> model.enhance -> formulation.build_sampler -> explicit dynamics updates
  -> WAV files + inference.json

evaluation.evaluate_results
  -> provenance validation -> MetricSuite.calculate -> metrics.csv + metrics.json
```

Pretraining and posttraining are independent algorithms over the same model.
Posttraining initialization imports pretrained weights; it does not execute the
pretraining method. Lightning pipelines only call the injected method and own
optimization, gradient clipping, logging and checkpoint hooks.

## DRC and CTC semantics

> **Release status (peer review):** the DRC and CTC implementations behind this
> section are withheld in this snapshot — `compute_drc`, `compute_ctc` and the
> CoF endpoint loss raise `NotImplementedError` with an explanatory message,
> and `cof train --training-stage posttrain` refuses the stage before creating a
> run directory. The configuration surface, method class, registries, and
> result records below remain part of the released contract; the algorithm is
> described in the paper (arXiv:2609.24651) and will be released upon
> acceptance.

The total objective is `drc.loss + ctc.weight * ctc.loss`. The reference
predictors are injected by name and supplied by the runner through EMA without
swapping live training parameters. CoF rollouts use the solver recorded in the
post-training configuration —
selectable per run through `--sampling-solver` or a preset, and inherited
from the pretraining run by default. SB-VE additionally
supports probability-flow ODE rollouts and inference; the finished run's
recorded solver is the recommended inference solver, and sampling with any
other solver warns.

## Extension points

1. **Backbone:** decorate an `nn.Module` with `BackboneRegister.register(name)`.
   Accept the model's `input_channels` construction argument and implement
   `forward(complex_inputs, time)`. Optional `training_preflight(config)` declares
   extra runtime requirements. Select its name and kwargs in the model config.
2. **Formulation:** implement `Formulation` with explicit dynamics, metadata and
   sampler factory; use `register_formulation(name)`. Add its preset directory for
   CLI configuration, or construct it directly in Python. Sampling implementation
   remains inside that formulation's package.
3. **Pretraining/posttraining method:** implement `TrainingMethod` — the base class
   abstracts `step`, and `from_config` is required in practice because the pipeline
   builder constructs methods through it — and register it in the appropriate
   method registry. `required_capabilities` checks compatibility before training.
   No stage pipeline edits are needed.
4. **Loss:** pass `loss_fn` to `BasePretraining` or `endpoint_loss` to
   `CoFPosttraining`. The default comes from the formulation or CoF definition.
   A trainable loss is exposed through `auxiliary_modules()` and registered once
   by the pipeline, included in its optimizer and Lightning state.
5. **Metric:** register a component with `output_names`, `higher_is_better`,
   `requires_reference`, and `calculate(ref_wav, deg_wav, sample_rate)`.
   `MetricSuite` rejects unknown names, duplicate outputs and nonfinite values.
   Validation and offline evaluation use this same suite. Calculators used with
   threaded offline evaluation must be thread-safe, as the built-in adapters are.

Application-specific registration modules must be imported before constructing a
model or resolving configuration. No implicit external plugin loading is performed.
An additional method's non-parameter state belongs in `state_dict/load_state_dict`;
its tensors/modules must be declared explicitly rather than hidden inside closures.

## Configuration and checkpoints

Tracked `config/` presets, ignored host `.config/cof.yml`, CLI overrides and
`runs/<run>/config.yml` remain the configuration lifecycle. `cof/config/` contains
the management code; it is not a replacement preset format. Pretraining defaults
to `pretrain.method: base`. CoF settings remain under `posttrain.cof`.

Validation metric selection is configurable in the presets:

```yaml
evaluation:
  metrics: [pesq, si_sdr]
  selection_metric: PESQ
```

The selection output must be enabled. Its component metadata determines min/max
selection, and `weights.default_test_model` records the selected export. Offline
metrics may be selected independently through the CLI.

A model export contains constructor metadata, transform parameters and model
weights. It excludes optimizer and method state. `from_checkpoint` resolves the
construction through a priority ladder: a self-describing checkpoint is
authoritative and a disagreeing surrounding config only triggers a warning that
keeps the checkpoint values; a legacy export without metadata falls back to the
requested config; a legacy export with no config to cross-check is rejected. A
Lightning checkpoint additionally contains the
optimizer, scheduler, EMA, progress and method identity/state. Resume selects `last.ckpt` when given
the run directory or its `checkpoints/` directory. An explicit `checkpoints/*.ckpt` file, including
`step=*.ckpt`, restores exactly that checkpoint. Missing files are rejected without fallback. It reuses persisted configuration, applies nested run overrides,
and does not reload the original pretraining initialization weights. EMA advances
once per optimizer step, including when gradient accumulation is active.

Run metadata records initialization provenance. Inference resolves the exact
selected model file and writes its hash and sampling protocol; evaluation validates
that attribution. Existing metric artifacts with a different requested metric
selection are rejected instead of silently reused.

## Validation

The refactor contract is enforced by the tracked test suite: `test/test_components.py`
(composition, checkpoint and extension behavior), `test/test_formulations.py`
(probability-path and solver contracts), `test/test_losses.py` (objectives and
gradient boundaries), `test/test_metrics.py` (metric registry and adapter),
`test/test_resolver.py` (configuration resolution), and `test/test_e2e.py`
(the real CLI deployment sequence on a synthetic dataset).
