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
2. `build_grounding.py`: builds offline biological grounding from mygene.info and CollecTRI signed TF-target edges, including shared-regulator and regulon summaries.
3. `build_traces.py`: uses a teacher model to rationalize known training labels, then prefilters, leak-filters, critic-filters, and logs rejected traces.
4. `train_sft_distill.py`: LoRA/QLoRA SFT of a student on accepted reasoning traces with explicit response-only loss masking.
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
- `pixi.toml`: locked environment, conservative thread env vars, and task wrappers.
- `paths.py`: repo layout; artifacts are addressed by run name under `../output/<stage>/<run>`.
- `config.py`: YAML config loader with inheritance and `--set key=value` overrides.
- `configs/sft/default.yaml`: SFT default for Qwen3-4B-Thinking.
- `configs/sft/v100_4b.yaml`: practical V100 SFT path.
- `configs/sft/h100_8b.yaml`: H100/NAIRR SFT path.
- `configs/sft/*screen*.yaml`: cheap 25-step or 75-step trace-source screens.
- `configs/sft/label_only_ablation.yaml`: empty-reasoning SynthPert control.
- `configs/grounding/default.yaml`: mygene, CollecTRI, context fields, and regulon-summary settings.
- `configs/grounding/no_collectri.yaml`: grounding ablation without CollecTRI.
- `configs/traces/default.yaml`: teacher/critic/sampling/split settings.
- `configs/traces/strict_*.yaml`: stricter teacher prompt/filter configs.
- `configs/grpo/default.yaml`, `h100.yaml`, `no_verifier.yaml`: GRPO model/reward/eval settings.
- `prompts/teacher/default.yaml`: default teacher rationale prompt, critic prompt, leak regexes.
- `prompts/teacher/strict_v2.yaml`: stricter evidence-constrained teacher/critic prompt.
- `prompts/student/default.yaml`: student system/question/answer format.
- `hf_import_shim.py`: avoids slow TRL import-time metadata scans on Lustre.
- `run_experiment.sbatch`: SLURM array for SFT/GRPO experiments with per-row trace runs.

## Current Local State

Checked on 2026-07-18 after commit `6bb6195` (`Track C strict trace SFT setup and response-only masking`).

- `../data/train.csv` and `../data/test.csv` exist.
- `../output/grounding/default/grounding.json` has been generated with the rescued CollecTRI static mirror. Last recorded stats:
  - `n_symbols=2623`
  - `n_annotated=2616`
  - `n_edges=48675`
  - `n_tfs=1187`
  - `n_rows=9518`
  - `n_rows_with_edge=37`
  - Static cache: `../output/grounding/default/cache/collectri_static.csv`
- `../output/traces` exists locally. Check individual run directories before assuming which trace sets are complete.
- No finished `../output/sft` or `../output/grpo` run was recorded in the prior context; recheck before launching dependent stages.
- Worktree note at this handoff: `README.md` has local/staged documentation edits and `run_experiment_nairr.sbatch` is untracked. Preserve both unless the user asks otherwise.

## Last Commit Summary

Commit `6bb6195` is a merge of PR #3 from `track-c-bootstrap`. Main changes:

- Adds strict trace-generation configs (`strict_smoke`, `strict_debug`, `strict_medium`, `strict_default`, `strict_large`) and `prompts/teacher/strict_v2.yaml`.
- Adds deterministic trace QA in `build_traces.py`: required gene mentions, configurable reject regexes, `prefilter` stats, and `rejects.jsonl` audit output.
- Makes SFT response-only masking explicit in `train_sft_distill.py` by constructing `input_ids`/`labels` directly and setting prompt labels to `-100`.
- Updates `task.py` so `think_prompt` does not duplicate `<think>` if the tokenizer chat template already emits it.
- Adds SFT screen/preflight configs for quick V100/H100 trace-source comparisons and label-only controls.
- Expands grounding context with shared regulator names, perturbation regulon genes, and target upstream TF summaries.
- Adds `hf_import_shim.py` and uses it before importing TRL to avoid slow Hugging Face metadata scans on Lustre.
- Pins Transformers to `>=4.56.2,<4.57`, sets conservative BLAS thread env vars, and simplifies GPU checks to import only Transformers in the default env.
- Updates `run_experiment.sbatch` to use seven-field experiment rows with per-experiment trace runs, override env vars, optional `USE_SRUN`, pixi cache placement, and CUDA override defaults.

## Current Runnable-State Notes

- SFT configs now exist under `configs/sft/`; the old missing-config warning is obsolete.
- GRPO configs now include `default`, `h100`, and `no_verifier`.
- CPU stages still require internet and API credentials where applicable. Use `OPENAI_API_KEY`, not `TEACHER_API_KEY`, for trace generation unless a config explicitly changes `teacher.api_key_env`.
- Strict trace outputs should use hyphenated run names such as `strict-smoke` or `strict-default`, while config names remain underscored (`strict_smoke`, `strict_default`).
- The static CollecTRI mirror is human-native. For mouse Track C, the repo uses upper-cased symbol matching; record this if reporting the method.

## Useful Commands

Run from repo root: `/expanse/lustre/projects/ddp412/i3gupta/tools/mlgenx/bioreason_sft`.

```bash
pixi run paths
pixi run data
env OPENBLAS_NUM_THREADS=1 pixi run grounding
OPENAI_API_KEY=... pixi run traces-smoke
OPENAI_API_KEY=... pixi run traces
pixi run label-traces
pixi run check-gpu-v100
pixi run sft-v100
pixi run -e h100 check-gpu-h100
pixi run -e h100 sft
pixi run grpo
pixi run results
```

Direct script equivalents:

```bash
python download_data.py
OPENBLAS_NUM_THREADS=1 python build_grounding.py --out default
python build_traces.py --out smoke --config smoke
python build_traces.py --out strict-smoke --config strict_smoke
python build_traces.py --out strict-default --config strict_default
python train_sft_distill.py --config v100_screen --traces strict-default --out screen-cleaned
python train_sft_distill.py --config h100_8b --traces default --out qwen8b
python train_grpo.py --config h100 --sft qwen8b --out qwen8b-grpo
```

## Future Chat Handoff Prompt

Use this when starting with a new LLM or future Codex chat:

```text
Read `.agents/context.md` first. This repo is for Kaggle MLGenX Bioreasoning
Challenge Track C. Continue from that context, inspect the current repo state,
and inspect existing output runs before launching dependent SFT or GRPO stages.
```
