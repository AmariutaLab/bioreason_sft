# Project Context: mlgenx/bioreason_sft

## Competition

Kaggle competition: MLGenX Bioreasoning Challenge, Track C.

URL: https://www.kaggle.com/competitions/ml-gen-x-bioreasoning-challenge-track-c/data

Task: mouse bone marrow-derived macrophage CRISPRi perturbation prediction. Given a
pair `(pert, gene)`, predict whether CRISPRi knockdown of `pert` makes `gene`
`up`, `down`, or `none`.

Local data files:

- `../data/train.csv`: 7705 rows, columns `id, pert, gene, label`
- `../data/test.csv`: 1813 rows, columns `id, pert, gene`
- Train labels: `none=4260`, `up=2359`, `down=1086`
- Unique train perturbations: 386
- Unique train target genes: 1570
- Unique test perturbations: 96
- Unique test target genes: 636

Metric implemented in `common.py`: mean of two rank-based AUROCs.

- DE-AUROC: true differential expression `(label != none)` ranked by `p_up + p_down`
- DIR-AUROC: true `up` vs `down` among true DE rows ranked by `p_up / (p_up + p_down)`
- Constant or heavily tied scores score about 0.5, so continuous probabilities matter.

Track C constraints captured by repo docs/prompts:

- Student model must be under 10B parameters.
- No tools, external models, or database lookup at inference.
- External biological resources are used only during offline data prep and distilled into model weights.

Kaggle CLI metadata checked on 2026-07-16:

- Track C deadline: `2026-07-22 07:00:00`
- Category: Community
- Reward: 2,000 USD
- Team count then: 45
- Competition data files then: `train.csv`, `test.csv`

## Repository Strategy

This repo implements reasoning distillation followed by GRPO, not a plain supervised baseline.

Pipeline:

1. `download_data.py`: downloads Kaggle CSVs into `../data`.
2. `build_grounding.py`: builds offline biological grounding from mygene.info and CollecTRI signed TF-target edges.
3. `build_traces.py`: uses a teacher model to rationalize known training labels, then leak-filters and critic-filters traces.
4. `train_sft_distill.py`: QLoRA SFT of a student on the accepted reasoning traces.
5. `train_grpo.py`: GRPO warm-started from the SFT adapter using format, correctness, and optional CollecTRI verifier rewards.
6. `common.py`: shared data loading, two-axis blocked split, local metric, temperature tuning, checkpoints, and wandb setup.
7. `task.py`: student prompt and answer-string invariant. It ensures `think_prompt` is a string prefix of `answer_prefix`, which is a string prefix of `train_text`, so inference reads A/B/C logits at the same token position optimized during training.

Key design decisions:

- Use a two-axis blocked validation split: perturbation and target gene are both disjoint between train and validation; off-diagonal rows are discarded to avoid leakage.
- Generate reasoning first, then read A/B/C logits in one forward pass for continuous rank scores.
- Grounding is train-time only; the student prompt contains no database text.
- CollecTRI is valuable because signed edges imply direction: knock down an activator -> target down; knock down a repressor -> target up.
- Trace generation uses SynthPert-style "rationalize the known label" rather than asking the teacher to predict from scratch.
- GRPO should be warm-started from SFT; RL alone does not add the reasoning primitives.

## Important Files

- `README.md`: detailed workflow and rationale.
- `pixi.toml`: locked environment and task wrappers.
- `paths.py`: repo layout; artifacts are addressed by run name under `../output/<stage>/<run>`.
- `config.py`: YAML config loader with inheritance and `--set key=value` overrides.
- `default.yaml`: appears to be an SFT config, but see current gaps below.
- `configs/grounding/default.yaml`: grounding sources and context fields.
- `configs/traces/default.yaml`: teacher/critic/sampling/split settings.
- `configs/grpo/default.yaml`: GRPO model/reward/eval settings.
- `prompts/teacher/default.yaml`: teacher rationale prompt, critic prompt, leak regexes.
- `prompts/student/default.yaml`: student system/question/answer format.
- `run_experiment.sbatch`: SLURM array for SFT/GRPO experiments.

## Current Local State

Checked on 2026-07-16.

- `../data/train.csv` and `../data/test.csv` exist.
- `../output/grounding/default/grounding.json` has been regenerated with the rescued CollecTRI static mirror:
  - `n_symbols=2623`
  - `n_annotated=2616`
  - `n_edges=48675`
  - `n_tfs=1187`
  - `n_rows=9518`
  - `n_rows_with_edge=37`
  - Static cache: `../output/grounding/default/cache/collectri_static.csv`
- The mygene RefSeq summary coverage is still low: `234/2623` symbols (`8.9%`).
- No `../output/traces`, `../output/sft`, or `../output/grpo` directories existed during inspection.

## Current Runnable-State Gaps

The checkout does not fully match the documented workflow.

- `configs/sft/` is missing, but `config.py` expects SFT configs at `configs/sft/<name>.yaml`.
- `README.md`, `pixi.toml`, and `run_experiment.sbatch` reference SFT configs such as `h100_8b`, `gemma4_e4b`, and `label_only_ablation`.
- Root-level `default.yaml` appears to be the SFT default config and likely belongs at `configs/sft/default.yaml`.
- `configs/grpo/default.yaml` exists, but docs/tasks reference GRPO configs such as `h100` and `gemma4_e4b` that are not present.
- `build_grounding.py` now prefers `https://rescued.omnipathdb.org/CollecTRI.csv`, caches it, parses `weight` as the signed edge, expands named complexes such as NFKB/AP1, and fails loudly on zero CollecTRI edges unless `--allow-no-edges` is passed.
- The static CollecTRI mirror is human-native. For mouse Track C, the repo uses upper-cased symbol matching; record this if reporting the method.

## Useful Commands

Run from repo root: `/expanse/lustre/projects/ddp412/i3gupta/tools/mlgenx/bioreason_sft`.

```bash
pixi run paths
pixi run data
env OPENBLAS_NUM_THREADS=1 pixi run grounding
OPENAI_API_KEY=... pixi run traces-smoke
OPENAI_API_KEY=... pixi run traces
pixi run check-gpu
pixi run sft
pixi run grpo
pixi run results
```

Direct script equivalents:

```bash
python download_data.py
OPENBLAS_NUM_THREADS=1 python build_grounding.py --out default
python build_traces.py --out smoke --config smoke
python build_traces.py --out default --config default
python train_sft_distill.py --config h100_8b --traces default --out qwen8b
python train_grpo.py --config h100 --sft qwen8b --out qwen8b-grpo
```

Before using direct training commands, fix the missing config files/directories.

## Future Chat Handoff Prompt

Use this when starting with a new LLM or future Codex chat:

```text
Read `.agents/context.md` first. This repo is for Kaggle MLGenX Bioreasoning
Challenge Track C. Continue from that context, inspect the current repo state,
and do not assume the documented pixi tasks are runnable until the config layout
is checked.
```
