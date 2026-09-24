<div align="center">
<img src="assets/corrective_forcing_banner.png" alt="Corrective Forcing" width="800">
</div>

**Corrective Forcing (CoF)** is a unified post-training paradigm for diffusion- and flow-based generative speech enhancement that adapts pretrained models to self-generated rollouts and corrects their predictions at both rollout-state and local-transition levels.

<div align="center">

[![Demo](https://img.shields.io/badge/Demo-CoF-4CAF50?logo=googlechrome&logoColor=white)](https://yorch233.github.io/CoF/)
[![GitHub](https://img.shields.io/badge/GitHub-Yorch233%2FCoF-181717?logo=github&logoColor=white)](https://github.com/Yorch233/CoF)
[![arXiv](https://img.shields.io/badge/arXiv-2609.24651-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.24651)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Yorch233%2FCoF-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/Yorch233/CoF)
[![ModelScope](https://img.shields.io/badge/ModelScope-Yorch233%2FCoF-6E37FA?logo=modelscope&logoColor=white)](https://www.modelscope.cn/models/Yorch233/CoF)

</div>

-------------------------

CoF is the official PyTorch implementation of [**Corrective Forcing: Unified Post-Training for Diffusions and
Flows in Generative Speech Enhancement**](https://arxiv.org/abs/2609.24651).

## Content

- [News](#news)
- [For Human](#for-human)
- [For Agent](#for-agent)
- [Skip This README](#skip-this-readme)
- [Overview](#overview)
  - [Introduction](#introduction)
  - [Backbone](#backbone)
  - [Project Architecture](#project-architecture)
  - [Documentation](#documentation)
- [Released Checkpoints](#released-checkpoints)
- [Manual Start](#manual-start)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

<a id="news"></a>

## 📰 News

- **2026-09-23** — Released the CoF codebase, pretrained and CoF post-trained checkpoints, together with the inference, dataset, and evaluation tooling. The complete CoF post-training pipeline is withheld during peer review and will be released upon acceptance.

<a id="for-human"></a>

## 👤 For Human

Copy and paste this prompt to your coding agent (Claude Code, Codex, OpenCode, etc.):

```text
Install and configure CoF by following the instructions here: https://raw.githubusercontent.com/Yorch233/CoF/refs/heads/main/docs/installation.md
```

<a id="for-agent"></a>

## 🤖 For Agent

Fetch the installation guide and follow it:

```bash
curl -s https://raw.githubusercontent.com/Yorch233/CoF/refs/heads/main/docs/installation.md
```

<a id="skip-this-readme"></a>

## ⚡Skip This README

We're past the era of reading docs. Just paste this into your agent:

```text
Read AGENTS.md and deploy CoF end to end: environment, dataset registration, stage-1 pretraining, CoF
post-training, few-step inference, and metrics.
```

<a id="overview"></a>

## 🔍 Overview

### Introduction

Diffusion and flow models for generative speech enhancement face a fundamental **training–inference mismatch**: during pretraining, models are optimized on analytical states sampled from prescribed probability paths, whereas inference recursively evaluates them on self-generated rollout states along discretized sampling trajectories. Prediction and discretization errors can therefore accumulate as inference proceeds.

**Corrective Forcing (CoF)** addresses this mismatch by adapting pretrained generative models directly to their own rollout distributions through two complementary components:

- **Dynamic Rollout Correction (DRC)** — trains the model on self-generated rollout states under dynamic
  sampling schedules, correcting clean-speech predictions toward the ground truth while exposing the model to
  diverse inference discretizations.
- **Counterfactual Transition Consistency (CTC)** — regularizes local evolution under transition-induced state
  deviations by aligning predictions after factual and locally corrected counterfactual transitions constructed
  from the same rollout state.

To apply the same post-training objective across different generative formulations, CoF expresses model outputs through a shared **clean-speech prediction parameterization**. CoF is validated on two representative generative formulations for speech enhancement, covering Schrödinger bridge
and flow matching:

- **SB-VE** — a Schrödinger bridge formulation that models stochastic probability transport between degraded
  and clean speech using a variance-exploding diffusion process.
- **OT-CFM** — an optimal-transport conditional flow-matching formulation that learns the vector field
  associated with a linear probability path, enabling deterministic transport from degraded toward clean speech.

### Backbone

The paper uses **NCSN++M** as the default backbone for both formulations, with approximately **27.8M parameters**
(27,756,314 in the default implementation). Both repository presets select it as `ncsnpp_base` with default
backbone settings.

### Project Architecture

The main source directories and supporting files are organized as follows:

```text
CoF/
├── cof/                  # Core Python package
│   ├── model.py          # Model, sampling, and checkpoint interface
│   ├── backbone/         # Neural networks and CUDA operators
│   ├── cli/              # Commands and interactive configuration
│   ├── config/           # Schemas and configuration resolution
│   ├── data/             # Paired data, synthesis, and STFT
│   ├── evaluation/       # PESQ, ESTOI, and SI-SDR metrics
│   ├── formulation/      # SB-VE and OT-CFM dynamics and samplers
│   ├── method/           # Pretraining and post-training interfaces
│   ├── pipeline/         # Workflows and artifact provenance
│   ├── training/         # Trainer, runtime, and callbacks
│   └── utils/            # Shared utilities
├── config/               # YAML defaults and formulation presets
├── docs/                 # Usage guides and detailed architecture
├── test/                 # Component, workflow, and safety tests
├── assets/               # README visual assets
├── AGENTS.md             # Agent deployment instructions
├── README.md             # Overview and quick start
└── pyproject.toml        # Dependencies and package metadata
```

### Documentation

- [Installation and runtime configuration](docs/installation.md)
- [Datasets, registration, and paired data creation](docs/datasets.md)
- [Training: pretraining and CoF post-training](docs/training.md)
- [Inference](docs/inference.md)
- [Metrics](docs/metrics.md)
- [Component architecture and extension points](docs/architecture.md)

<a id="released-checkpoints"></a>

## 📦 Released Checkpoints

The checkpoint collection uses the **NCSN++M backbone (approximately 27.8M parameters)** throughout.
It contains twelve models: a pretrained model and a CoF post-trained model for each combination of
two formulations (SB-VE and OT-CFM) and three benchmark datasets.

Browse the model repositories below to select a dataset, formulation, and training stage:

- Hugging Face — [View All →](https://huggingface.co/Yorch233/CoF)
- ModelScope — [View All →](https://www.modelscope.cn/models/Yorch233/CoF)

| Dataset | Task | Formulations | Stages |
| --- | --- | --- | --- |
| Voicebank+Demand | denoising (8 DEMAND + 2 synthetic noises, SNR 0–17.5 dB) | SB-VE, OT-CFM | pretraining, CoF |
| WSJ0+WHAM | denoising with real-world ambient noise (SNR −6 to 14 dB) | SB-VE, OT-CFM | pretraining, CoF |
| WSJ0+Reverb | dereverberation (synthetic RIRs, T60 0.4–1.0 s) | SB-VE, OT-CFM | pretraining, CoF |

Models are organized as `<dataset>/<formulation>/<stage>/`. Each directory contains a self-describing
`.safetensors` file and its `config.yml`. Weight filenames follow
`<dataset>_<formulation>[_cof].safetensors`, with `_cof` identifying post-trained models
(for example, `voicebank_sbve_cof.safetensors`).

The model cards document file SHA-256 hashes and the recommended inference solver. For CoF checkpoints,
this is the solver used to generate the post-training rollouts. See [Manual Start](#manual-start) for
environment setup, dataset registration, and evaluation.

<details>
<summary><b>Use a downloaded checkpoint</b></summary>

A downloaded checkpoint runs directly through `--ckpt` — no run directory or training required:

```bash
uv run cof inference \
  --ckpt /path/to/voicebank+demand/sb-ve/CoF/voicebank_sbve_cof.safetensors \
  --dataset voicebank \
  --split test \
  --num-steps 4
```

For loose audio files, use `cof enhance --ckpt ...`. Both commands read model settings and sampling defaults
from the checkpoint's embedded metadata.

</details>

<a id="manual-start"></a>

## 🚀 Manual Start

Clone the repository — every command below runs from its root:

```bash
git clone https://github.com/Yorch233/CoF.git
cd CoF
```

> The examples assume the registered dataset ID `voicebank` and the default SB-VE deployment; replace
> paths and run names with values for your deployment.

- **Train from scratch:** complete [Step 1](#step-1), [Step 2](#step-2), [Step 3](#step-3),
  [Step 4](#step-4), [Step 5](#step-5), [Step 6](#step-6), and [Step 7](#step-7).
  **This full workflow is not supported in the current snapshot**
  because CoF post-training is withheld during peer review.
- **Post-train from the released pretrained checkpoints:** complete [Step 1](#step-1), [Step 2](#step-2),
  and [Step 3](#step-3), then [Step 5](#step-5), [Step 6](#step-6), and [Step 7](#step-7) using a pretrained
  model from [Released Checkpoints](#released-checkpoints). **This workflow is not supported in the current snapshot**
  because CoF post-training is withheld during peer review.
- **Evaluation using released checkpoint:** complete [Step 1](#step-1), [Step 2](#step-2), and [Step 3](#step-3),
  download a model from [Released Checkpoints](#released-checkpoints), then complete [Step 6](#step-6) and [Step 7](#step-7).

<a id="step-1"></a>

### Step 1 — Set up the environment

Create the project environment and install dependencies from `pyproject.toml`:

```bash
uv sync
```

👉 [Installation guide](docs/installation.md) — host prerequisites, environment setup, and CUDA/JIT operators.

<a id="step-2"></a>

### Step 2 — Prepare a paired dataset

CoF expects aligned WAV files under `{train,valid,test}/{clean,noisy}`. Register an existing paired corpus:

```bash
uv run cof dataset add --id voicebank --path /path/to/Voicebank+DEMAND --select
uv run cof dataset list
```

<details>
<summary><b>Create a paired dataset from separate corpora</b></summary>

When only separate clean and noise corpora exist, synthesize the pairs first and register the export:

```bash
uv run cof dataset create --task enhancement \
  --clean wsj0 /path/to/wsj0 --noise wham /path/to/wham \
  --output-dir /path/to/wsj0-wham
uv run cof dataset add --id wsj0-wham --path /path/to/wsj0-wham --select
```

Supported clean sources are `vctk`, `wsj0`, and `timit`; supported noise sources are `none`, `chime`, `qut`, and
`wham`.

</details>

👉 [Dataset guide](docs/datasets.md) — paired-data layout, registration, task matrix, synthesis parameters, and split rules.

<a id="step-3"></a>

### Step 3 — Configure the host

Open the interactive wizard to configure the dataset selection, devices, logging, and output directories:

```bash
uv run cof config
```

The wizard writes host settings — dataset registry and selection, precision, GPU selection, logger, logging and
checkpoint intervals, and the run and result roots — to the ignored `.config/cof.yml`. The stage and formulation
come from the CLI, not from this file.

👉 [Installation and runtime configuration](docs/installation.md) — host settings and configuration checks.

👉 [Training guide](docs/training.md) — stage presets, configuration inheritance, and per-run overrides.

<a id="step-4"></a>

### Step 4 — Pretrain the generative model

Train an SB-VE generative model from scratch on the registered `voicebank` dataset:

```bash
uv run cof train \
  --training-stage pretrain \
  --formulation SB-VE \
  --dataset voicebank
```

> The command displays the effective configuration for confirmation. Add `--yes` to skip the prompt
> after reviewing the configuration, or for unattended runs.

<details>
<summary><b>Pretraining configuration and run artifacts</b></summary>

This stage is epoch-based and trains from scratch. `--formulation` selects the generative formulation —
`SB-VE` or `OT-CFM` — and loads the corresponding `config/<formulation>/pretrain.yml` preset. Like the training
stage, it is declared per invocation, not through host configuration. The choice is made once here:
post-training takes no formulation flag and always continues the formulation of the `--pretrain-run` it points
at. The run creates
`runs/cof_sbve_pretrain_MMDDhhmm/` and exports `model_valid_pesq=*.safetensors`, `model_valid_loss=*.safetensors`,
and `model_last.safetensors` as EMA weights.

</details>

👉 [Training guide](docs/training.md) — pretraining configuration, validation, and run artifacts.

<a id="step-5"></a>

### Step 5 — Run CoF post-training

Select the pretraining run whose configuration and weights will initialize CoF post-training:

In this release, the CLI command below prints the peer-review release notice and exits without starting training or creating a run directory.

```bash
uv run cof train \
  --training-stage posttrain \
  --pretrain-run runs/cof_sbve_pretrain_MMDDhhmm \
  --dataset voicebank
```

> When post-training is available, add `--yes` to skip configuration confirmation after reviewing the
> configuration, or for unattended runs. This flag does not enable post-training in the current snapshot.

<details>
<summary><b>Post-training configuration and parameter inheritance</b></summary>

Post-training is step-based (4 000 steps in the released preset) and **requires** `--pretrain-run`: the stage
reads the pretraining run's `config.yml`, inherits every model- and protocol-defining parameter (formulation,
backbone and kwargs, formulation kwargs, sampling step count and skip type), and resolves the initialization
weights from the run's exports. These inherited parameters cannot be overridden on the CLI. The solver is
an exception: `--sampling-solver` selects the post-training rollout and validation solver, defaulting to the
pretraining run's solver. In this snapshot the post-training stage is refused with the release-status notice
above. Every configuration surface it documents is already present.

</details>

👉 [Training guide](docs/training.md) — CoF post-training, DRC schedule and CTC controls, resuming, and GPU-memory options.

To export an existing completed run, use `cof run export`. For example, export the pretraining run from
Step 4, replacing the placeholder with its exact recorded run name:

```bash
uv run cof run export cof_sbve_pretrain_MMDDhhmm \
  --output-dir /path/to/export
```

This writes `voicebank_sbve.safetensors`, `config.yml`, and `export.json` for the example run. The same command
accepts an existing completed post-training run. Its model filename includes the `_cof` suffix.
Use the exported `.safetensors` path with `--ckpt` in Step 6.

<a id="step-6"></a>

### Step 6 — Enhance the test set

Enhance the registered test split using a locally downloaded checkpoint from
[Released Checkpoints](#released-checkpoints). Alternatively, set the `--ckpt` value to the model path exported
in Step 5. Checkpoints are not downloaded automatically.

```bash
uv run cof inference \
  --ckpt /path/to/voicebank_sbve_cof.safetensors \
  --dataset voicebank \
  --split test \
  --num-steps 4
```

<details>
<summary><b>Infer directly from an existing run</b></summary>

Use `--run` instead of `--ckpt` to resolve a completed run's default model. For example, using the
pretraining run from Step 4:

```bash
uv run cof inference \
  --run cof_sbve_pretrain_MMDDhhmm \
  --dataset voicebank \
  --split test \
  --num-steps 4
```

</details>

<details>
<summary><b>Inference artifacts and sampling step counts</b></summary>

With `--ckpt`, inference writes `results/<checkpoint-stem>/<solver>_N=<steps>/`. With `--run`, it uses
`results/<run-name>/<solver>_N=<steps>/`. Each variant contains one WAV per paired test utterance plus an
`inference.json` manifest that records the dataset, split, run, model SHA-256, sampler, step count, and seed.
CoF targets robust enhancement across different numbers of sampling steps, including few-step inference, so
evaluate the step counts relevant to your deployment: the evaluation
protocol of this release covers `--num-steps 1`, `4`, and `16` for SB-VE, and `1` and `4` for OT-CFM.

</details>

👉 [Inference guide](docs/inference.md) — sampling protocols, result variants, manifests, and provenance checks.

<a id="step-7"></a>

### Step 7 — Evaluate the results

Evaluate the enhanced audio in the result directory printed by Step 6. For the downloaded checkpoint example
with the default result root:

```bash
uv run cof metric --dir results/voicebank_sbve_cof/SB_SDE_Solver_N=4
```

This computes the three intrusive core metrics (PESQ, ESTOI, SI-SDR), all through TorchMetrics, and writes
`metrics.csv` and `metrics.json` into the result directory. Run-linked evaluation refuses to score a result
whose manifest does not match the run it is attributed to.

👉 [Metrics guide](docs/metrics.md) — supported metrics, evaluation modes, and output files.

<a id="citation"></a>

## 📖 Citation

If you find CoF useful in your research, please cite our paper:

```bibtex
@misc{yao2026corrective,
  title={Corrective Forcing: Unified Post-Training for Diffusions and Flows in Generative Speech Enhancement},
  author={Yao, Qing and Gao, Lijian and Mao, Qirong},
  journal={arXiv preprint arXiv:2609.24651},
  year={2026}
}
```

<a id="acknowledgements"></a>

## 🙏 Acknowledgements

We thank the official implementations this framework builds on:

- [RSB](https://github.com/Yorch233/RSB) — the implementation of the SB-VE formulation for speech enhancement.
- [FlowSE](https://github.com/seongq/flowmse) — the implementation of the OT-CFM formulation included as the
  second deployment path, following FlowSE's reverse time convention.
- [StoRM](https://github.com/sp-uhh/storm) — the dataset construction and data-processing conventions.

<a id="license"></a>

## 📄 License

CoF is licensed under the [Apache License 2.0](LICENSE).
