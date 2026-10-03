#!/bin/zsh
# #1476: when 64 corpus buckets are done, draw the per-sense gate sample, label it with Opus, score it. Log: gate_chain.log
# Run from the working directory that holds data/; needs CLAUDE_API_KEY, DATABRICKS_WAREHOUSE_ID and SENSES_OUT (corpus_run.py's output folder).
# zsh gate_chain.sh >> gate_chain.log 2>&1 &
SENSES_OUT=${SENSES_OUT:-out}
until [ $(ls $SENSES_OUT/b*.jsonl 2>/dev/null | wc -l) -ge 64 ]; do sleep 60; done
echo "$(date '+%H:%M') 64 buckets; sampling"
uv run --quiet --with httpx python gate.py sample || exit 1
uv run --quiet --with anthropic python gold_label.py --in data/gate_works.jsonl --out data/gate_labels.jsonl || exit 1
uv run --quiet python gate.py score
echo "$(date '+%H:%M') GATE_DONE"
