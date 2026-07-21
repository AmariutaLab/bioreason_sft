#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in this shell/job before running}"

TRACE_PIXI_ENV="${TRACE_PIXI_ENV:-default}"
TRACE_RUN="${TRACE_RUN:-norules-ungrounded-o4mini-noval-fulltrain-clean2400}"
SEED_TRACES="${SEED_TRACES:-../output/traces/norules-ungrounded-o4mini/traces.jsonl}"
WORKERS="${WORKERS:-2}"

echo "[trace] run=$TRACE_RUN"
echo "[trace] seed accepted traces=$SEED_TRACES"
echo "[trace] max_tokens=2400 workers=$WORKERS"

pixi run -e "$TRACE_PIXI_ENV" python build_traces.py \
  --config strict_medium_ungrounded_o4_mini \
  --out "$TRACE_RUN" \
  --no-val \
  --no-grounding \
  --extend-traces "$SEED_TRACES" \
  --max-tokens 2400 \
  --set prompts=teacher/norules_ungrounded \
  --set sampling.balance_classes=false \
  --set sampling.n=1000000000 \
  --set workers="$WORKERS"

pixi run -e "$TRACE_PIXI_ENV" python clean_trace_jsonl.py \
  --input "../output/traces/$TRACE_RUN/traces.jsonl" \
  --out-run "${TRACE_RUN}-clean" \
  --min-critic-score 4

echo "[trace] raw:   ../output/traces/$TRACE_RUN/traces.jsonl"
echo "[trace] clean: ../output/traces/${TRACE_RUN}-clean/traces.jsonl"
