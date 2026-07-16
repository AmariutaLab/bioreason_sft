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

## Install

```bash
conda env create -f environment.yml
conda activate bioreason
python -m ipykernel install --user --name bioreason --display-name "bioreason"
```

For this Expanse workspace, a validated CPU/code-check Conda env is available at:

```bash
conda activate /expanse/lustre/projects/ddp412/zxu6/mlgenx/envs/bioreason
```

For GPU training, verify the active env on a GPU node before submitting long jobs:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

The default Conda path uses standard Transformers + PEFT fp16 LoRA, so Expanse
V100 GPUs are usable with the smaller `v100_4b` config. 4-bit / Unsloth variants
are optional and may need newer GPUs or non-Conda packages.

If Conda solving is slow, use `mamba env create -f environment.yml` with the same
file. Keep package changes in `environment.yml` so notebooks, login-node data
prep, and SLURM jobs use the same environment.

On Expanse, if Conda reports `CondaVerificationError` for packages under
`~/miniconda3/pkgs`, use a fresh package cache on scratch instead of the default
cache:

```bash
SCRATCH_ROOT=${SCRATCH:-/scratch/$USER/job_${SLURM_JOB_ID}}
mkdir -p "$SCRATCH_ROOT/conda_pkgs" "$SCRATCH_ROOT/envs"
CONDA_PKGS_DIRS="$SCRATCH_ROOT/conda_pkgs" conda env create \
  -p "$SCRATCH_ROOT/envs/bioreason" \
  -f environment.yml
conda activate "$SCRATCH_ROOT/envs/bioreason"
```

On Expanse, the automated setup path is:

```bash
sbatch -A csd832 -p compute -t 04:00:00 -N 1 -n 1 -c 4 --mem=32G scripts/setup_env.sbatch
sbatch -A csd832 -p gpu-debug --gpus=1 -N 1 -n 1 -c 4 --mem=32G \
  --dependency=afterok:<setup_job_id> scripts/gpu_smoke.sbatch
```

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
python make_submission.py --stage sft --run qwen8b
python validate_submission.py ../output/submissions/sft-qwen8b.zip
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
sbatch -A csd832 run_experiment.sbatch              # default array runs the first V100 SFT
sbatch -A csd832 --array=1 run_experiment.sbatch    # run one other experiment
sbatch -A csd832 --export=ALL,TRACES_RUN=v2 run_experiment.sbatch
```

Run the CPU stages (download / grounding / traces) once on the login node first —
they need internet, not a GPU. Update the `module load` / `conda activate` lines
and `--partition` for your cluster.

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
the metric half even SOTA only reaches ~0.65–0.73 on.

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
