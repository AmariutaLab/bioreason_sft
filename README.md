# bioreason_sft — Track C: reasoning distillation → GRPO

CRISPRi perturbation prediction in mouse BMDMs. `(pert X, gene Y) → up / down / none`,
scored as the mean of two rank-based AUROCs (DE and DIR).

## Layout

```
mlgenx/
├── bioreason_sft/              # code + configs + prompts (this dir)
│   ├── configs/<stage>/*.yaml  # what to run
│   ├── prompts/<kind>/*.yaml   # all prompt text
│   ├── run_experiment.sbatch
│   └── pipeline.ipynb          # notebook version of the CLI
├── data/                       # train.csv, test.csv
└── output/
    ├── grounding/<run>/grounding.json
    ├── traces/<run>/traces.jsonl
    ├── sft/<run>/{adapter,checkpoints,metrics.json,resolved_config.json}
    └── grpo/<run>/{adapter,best,metrics.json,resolved_config.json}
```

**Artifacts are addressed by run name, never by path**: `--traces v2` reads
`output/traces/v2/traces.jsonl`; `--out qwen8b` writes `output/sft/qwen8b/`.

Keep `data/` and `output/` outside the Git repo. They are local Kaggle data and
generated artifacts, not source files.

## Install (pixi)

```bash
cd bioreason_sft
pixi install                 # default env: V100-compatible Torch/PEFT path
pixi run check-gpu-v100      # run this ON A V100 GPU NODE
```

`pixi.lock` pins **both** the conda and PyPI sides, which matters because this
stack breaks exactly at that boundary. The default env is intentionally
V100-compatible for `csd832`; the optional `h100` env adds Unsloth/bitsandbytes
for later NAIRR/H100 access.

Environments: `default`/`v100` (CLI), `h100` (+Unsloth/bitsandbytes), `nb`
(+jupyter), `wandb` (+logging), `full` (all).

```bash
pixi run -e nb notebook          # jupyter lab pipeline.ipynb
pixi run -e h100 check-gpu-h100  # later, on H100/NAIRR only
pixi shell                       # drop into the env
```

The V100 path uses standard Transformers + PEFT fp16 LoRA. The H100 path can use
4-bit/Unsloth accelerators, but that should wait until the `sdp147`/NAIRR access
issue is resolved.

### Tasks

`pixi task list` shows them all. The pipeline, in order:

| Task | Where | What |
| --- | --- | --- |
| `pixi run data` | login node | download Kaggle data → `../data` |
| `pixi run grounding` | login node | mygene + CollecTRI → `../output/grounding/default` |
| `pixi run traces-smoke` | login node | 30 traces — **run this and read them first** |
| `pixi run traces` | login node | full generation (~1–3 h, resumable) |
| `pixi run label-traces` | anywhere | deterministic blocked-train fallback traces |
| `pixi run sft-v100` | **V100 GPU** | practical 4B baseline with fallback traces |
| `pixi run -e h100 sft` | **H100 GPU** | later H100/NAIRR path |
| `pixi run sft-control` | **GPU** | the label-only control |
| `pixi run grpo` | **GPU** | RL on top of the SFT adapter |
| `pixi run results` | anywhere | leaderboard of all local runs |
| `pixi run submit` | login node | `sbatch` the experiment array |

Tasks declare dependencies (`traces` → `grounding` → `data`), so `pixi run
traces-smoke` on a clean checkout does the whole cheap chain. `pixi run
pipeline-smoke` is that chain explicitly.

Tasks are convenience wrappers for the common case — for anything custom, call
the scripts directly with `--config` / `--set` (see below).

## The config system

**Source code never changes for an experiment.** Three levers, in order of preference:

**1. Pick a config.** `--config h100_8b` loads `configs/sft/h100_8b.yaml`.

**2. Write a new config that inherits.** Override only what changes:

```yaml
# configs/sft/my_experiment.yaml
extends: default          # deep-merged: everything else inherited
lora:
  r: 64                   # dropout, target_modules etc. still come from default
```

**3. One-off overrides.** `--set key.path=value`, YAML-parsed:

```bash
--set lora.r=64 --set train.lr=1e-5 --set model.load_in_4bit=false \
--set lora.target_modules='[q_proj, v_proj]'
```

Every run writes `resolved_config.json` next to its outputs — the *fully merged*
config that actually produced those artifacts. The yaml on disk may change later;
the snapshot won't.

### What lives where

| Experiment | Edit |
| --- | --- |
| **Teacher prompts** (wording, rules, critic rubric, leak patterns) | `prompts/teacher/*.yaml` |
| **Teacher model / API / sampling / critic bar** | `configs/traces/*.yaml` |
| **Grounding sources** (mygene, CollecTRI, what goes in the context block) | `configs/grounding/*.yaml` |
| **Student prompts** (system, question, answer format) | `prompts/student/*.yaml` |
| **Student model, LoRA, training, eval** | `configs/sft/*.yaml` |
| **GRPO rewards, KL, generations, eval cadence** | `configs/grpo/*.yaml` |

To try a new teacher prompt: copy `prompts/teacher/default.yaml` →
`terse.yaml`, edit, then `--set prompts=teacher/terse`. No code touched.

## Choosing a base model

Track C caps the student at **<10B parameters**. Practical configs currently
ship for the V100-accessible 4B path and the later H100 path:

```bash
python train_sft_distill.py --config v100_4b --traces label-default --out qwen4b-v100
python train_sft_distill.py --config h100_8b --traces default --out qwen8b
```

| Model | Params | Eligible | Notes |
| --- | --- | --- | --- |
| **Qwen3-8B** | 8.2B dense | yes | **default pick.** Most knowledge capacity in budget; mature fine-tuning ecosystem |
| Qwen3-4B-Thinking | 4B | yes | organizers' baseline; fits a T4 |
| Gemma 4 E4B | ~4.5B effective (~6B total) | yes | Apache 2.0, native thinking; but *edge* tier (Per-Layer Embeddings) |
| Gemma 4 E2B | ~2.3B effective | yes | too small for a knowledge-bound task |
| Gemma 4 26B A4B | 25.2B total / 3.8B active | **probably not** | MoE. "<10B" almost certainly means *total* — ask the organizers before relying on it |
| Gemma 4 12B / 31B | 12B / 30.7B | no | over the cap |

**Why Qwen3-8B by default:** this task is *knowledge-bound* — the bottleneck is
knowing what `Sbno2` does in a macrophage, not reasoning capacity. Under the
superficial-alignment hypothesis the knowledge must already be latent in
pretraining for traces to activate it, so parameters-for-facts is the thing to
buy. Gemma's E2B/E4B are built for phones and trade exactly that away.

Gemma A/B configs can be added later, but the immediate goal is to get one
robust baseline through the full submission pipeline before expanding the sweep.

**Kaggle model mounts are irrelevant here.** They exist for Kaggle notebooks with
internet off. On SLURM, HF is the same weights by a faster path.

## Kaggle credentials

Kaggle has **two** credential styles; `download_data.py` accepts either.

**New (token)** — kaggle.com/settings/api → *"Generate New Token"*:
```bash
export KAGGLE_API_TOKEN=<token>
# or on disk:
mkdir -p ~/.kaggle && echo <token> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

**Legacy (json)** — same page → *"Legacy API Credentials"* → *"Create Legacy API Key"*:
```bash
~/.kaggle/kaggle.json        # chmod 600
# or:
export KAGGLE_USERNAME=... KAGGLE_KEY=...
```

The script discovers whichever you have, exports the env vars the downstream
tools expect (so a file-only setup works without exporting anything), and tries
kagglehub → kaggle CLI → direct HTTP in order. `KAGGLE_CONFIG_DIR` is honoured.

> **You must also ACCEPT THE COMPETITION RULES on the website** with the same
> account. Rules not accepted is the cause of virtually every 403.

If the cluster blocks egress, download locally and `scp -r data/ cluster:mlgenx/`.

## Running a full experiment

```bash
cd bioreason_sft

# 0. data (CPU) — see "Kaggle credentials" above
python download_data.py

# 1. grounding (CPU, ~5 min, no API key)
python build_grounding.py --out default

# 2. traces — SMOKE TEST FIRST. 30 rows costs cents.
export TEACHER_API_KEY=...
python build_traces.py --out smoke --config smoke
#    then READ the traces (see "Trace QA" below) before spending on:
python build_traces.py --out default --config default

# fallback if no teacher key yet: deterministic blocked-train label traces
python build_label_traces.py --out label-default --config v100_4b

# 3. SFT (GPU, ~1-2 h)
python train_sft_distill.py --config v100_4b --traces label-default --out qwen4b-v100

# 4. GRPO (GPU, several h)
python train_grpo.py --config h100 --sft qwen8b --out qwen8b-grpo

# 5. submission
python make_submission.py --stage sft --run qwen4b-v100
python validate_submission.py ../output/submissions/sft-qwen4b-v100.zip
```

### Resume

```bash
python train_sft_distill.py --traces default --out qwen8b --resume
# [resume] checkpoint-250: stopped at epoch 2.00, step 250/375
```

HF `Trainer` is the resume mechanism — a checkpoint carries optimizer state, LR
schedule position, RNG state and `global_step`/`epoch`. `--resume` finds the
newest *complete* checkpoint (partial writes without `trainer_state.json` are
ignored). **Resume needs identical data and args**; the script warns if
`max_steps` disagrees. `--resume` is safe on a fresh run, so requeued SLURM jobs
self-heal.

wandb (`--set wandb.enabled=true`) is for configs and curves, **not** resume.
`--wandb-id <id>` rejoins the same logging run so curves connect.

## SLURM

Edit the `EXPERIMENTS` array in `run_experiment.sbatch` — one line per
experiment, `"STAGE|SFT_CONFIG|SFT_RUN|GRPO_CONFIG|GRPO_RUN|EXTRA"`:

```bash
EXPERIMENTS=(
  "sft|v100_4b|qwen4b-v100||| "
  "sft|label_only_ablation|abl-labelonly||| "
  "sft+grpo|h100_8b|qwen8b|h100|qwen8b-grpo|"
  "sft+grpo|h100_8b|qwen8b-r64|h100|qwen8b-r64-grpo|--set lora.r=64 --set lora.alpha=128"
)
```

```bash
sbatch -A csd832 run_experiment.sbatch              # default V100 baseline
sbatch -A csd832 --array=1 run_experiment.sbatch    # label-only control

# later, after NAIRR access:
sbatch -A sdp147 -p nairr-gpu-shared --gpus=h100:1 --array=2-3 \
  --export=ALL,PIXI_ENV=h100,CHECK_TASK=check-gpu-h100,TRACES_RUN=default \
  run_experiment.sbatch
```

Run the CPU stages (download / grounding / traces) once on the login node first —
they need internet, not a GPU. The script activates the pixi env via
`pixi shell-hook` (no conda), so compute nodes get the same `pixi.lock` the login
node used. Update `--partition`, the `module load cuda` line, and `PIXI_HOME` for
your cluster. It passes `--resume` unconditionally, so requeued jobs self-heal.

## Notebook

`pipeline.ipynb` runs the same stages with inspection cells between them
(coverage stats, trace samples, val reasoning, a GRPO val-curve plot, and a
run-comparison table).

## Design decisions worth knowing

**Two-axis blocked split.** The real split is disjoint on the perturbation axis
(80/10/10) *and* the gene axis (60/20/20) — no gene appears in more than one
split. So val = rows where *both* are held out; train = rows where *neither* is;
off-diagonal rows are **discarded** (~30%). A one-axis or random split leaks and
flatters your score.

**The string invariant.** `think_prompt ⊂ answer_prefix ⊂ train_text`, all built
by concatenating onto one `chat_head`. The inference prefix is a literal string
prefix of the training text, so the logit read lands exactly where training
optimized. `Task.resolve_letter_ids()` verifies at runtime that greedy BPE
doesn't merge `>` + `A` into one `>A` token — it tries separators and picks one
that provably works instead of assuming.

**Continuous scores, not letters.** The metric is rank-based AUROC. We generate
the `<think>` block, then read the A/B/C logits in one forward pass, then tune
temperature on val. The official baseline takes one sample → one letter → a fixed
probability pair, so its scores have few distinct values and mostly tie at ~0.5.
(A constant score vector scores *exactly* 0.5 — verified.)

**Grounding is train-time only.** Track C forbids tools at *inference*, not during
data prep. SynthPert found injecting pathway text at inference *hurt* by 0.07
AUROC via context saturation — the opposite mechanism from distilling grounding
into weights. The student never sees a database.

**CollecTRI is the highest-value source** because it's *signed*: knock down an
activator → target down; knock down a repressor → target up. That's DIR-AUROC,
the metric half even SOTA only reaches ~0.65–0.73 on. We fetch it straight from
OmniPath's REST API with `requests` rather than via `decoupler` — decoupler
depends on numba, whose resolver backtracks to numba 0.53.1 (no Python 3.12
wheel, and its sdist refuses to build: *"only versions >=3.6,<3.10 are
supported"*). We used decoupler for exactly one download, so we do the download.
The optional library path still exists: `pixi run -e grounding-decoupler
grounding` with `--set sources.collectri.method=decoupler`.

Two cleanups happen on the way in: rows with **ambiguous signs** (both or neither
stimulation/inhibition) are dropped, because an edge without a direction is worse
than no edge here; and **protein complexes** (`FOS_JUN`, `FOSL1_JUNB`) are split
into member TFs, since a complex name never matches a single perturbed gene
symbol and would silently cost coverage.

**Approach 2 (rationalize the known label)** means the teacher's own accuracy
stops mattering: o4-mini scored ~52% on this task, yet its traces trained an 8B
student to ~89% — the student beats its teacher, because the knowledge is already
latent in pretraining and traces just activate the reasoning pattern.

**The leak filter** is the piece SynthPert doesn't detail. Handing the teacher the
answer invites backward reasoning ("since the answer is B..."), which teaches the
student nothing. 10 regex patterns, hard-rejected before the critic runs.

**GRPO warm-starts from SFT** — required, not optional. RL alone doesn't add new
reasoning priors; SFT on traces adds the primitives RL then explores.

## The three things that will bite you

1. **Class collapse.** `none` is 55% of rows → "always answer C" maximizes naive
   reward → DIR-AUROC 0.5. Handled by a balanced buffer + inverse-frequency
   reward weights (`rewards.correctness.class_balance`).
2. **Calibration collapse.** Correctness-only RL sharpens toward argmax; scores
   tie; AUROC craters **while reward climbs**. Watch `val/score`, not the reward.
   Guards: `grpo.beta` (KL) and the val callback, which saves the best adapter to
   `output/grpo/<run>/best/`. Use *that*, not the final one.
3. **Verifier noise.** `validate_verifier()` prints CollecTRI's own direction
   accuracy on covered rows and auto-disables below
   `rewards.verifier.min_direction_acc`. A noisy reward is worse than none.

## Trace QA (do not skip)

After the smoke run, read 5–10 traces. You're checking for the SynthPert
signature: does it reason about **specific gene-level function** ("Rpf2 matures
the 60S subunit, so its loss slows ribosome biogenesis") or hand-wave about
categories ("both are involved in the stress response")? The former generalizes;
the latter correlated with *wrong* answers in SynthPert's audit.

Also check the `none` traces specifically — they're 55% of the label space and
the hardest to write well. If they're vacuous ("no known link"), fix
`prompt_none` in `prompts/teacher/*.yaml`.

Watch the keep-rate breakdown. High `leak` → tighten the rules. ~100% `lowscore`
→ `--set critic.min_score=4` or a stronger teacher. A *low keep rate is not a
bug*: SynthPert trained on ~2% of data and beat full-data runs.

## Ablations worth reporting

You're publishing regardless of rank, and negative results here are publishable.

| Run | Purpose |
| --- | --- |
| zero-shot base | honest floor |
| `--config label_only_ablation` | the SynthPert control (~0.59 DE on human lines) |
| trace SFT | the treatment |
| traces built with `--set sources.collectri.enabled=false` | isolates CollecTRI |
| traces with `--set critic.enabled=false` | isolates quality filtering |
| `+ GRPO` | isolates RL's marginal gain |
| `--config no_verifier` | isolates the soft verifier |
| CollecTRI rule alone | non-LLM baseline on covered rows |
