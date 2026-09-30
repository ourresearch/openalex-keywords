#!/bin/zsh
# Relabel driver (#1322, 2026-09-25): 100K-request waves on the Batches API (prompt v3), each collected before the next is submitted, then synchronous retry of
# refusals / errors. Resumable: judge_batch.py skips ids already collected or submitted. Log: data/driver.log.
set -e
: "${CLAUDE_API_KEY:?}"  # env: CLAUDE_API_KEY (or ANTHROPIC_API_KEY), DATABRICKS_HOST / DATABRICKS_TOKEN (CLI auth), DATABRICKS_WAREHOUSE_ID; export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-$CLAUDE_API_KEY}
cd "$(dirname "$0")"   # code/teacher: prompt_v3.py, judge_batch.py, data/, results/, batches/
J=(uv run -q --with anthropic python judge_batch.py --prompt prompt_v3 --tag v3 --sample data/sample.jsonl)
for wave in $(seq 1 11); do
  echo "$(date '+%F %T') wave $wave submit" 
  "${J[@]}" submit --limit 100000 | tail -2
  while true; do
    st=$("${J[@]}" poll 2>&1) || { echo "poll failed: $st" | tail -3; sleep 120; continue; }
    if ! echo "$st" | grep -q "in_progress\|canceling"; then echo "$st" | tail -1; break; fi
    sleep 120
  done
  echo "$(date '+%F %T') collected: $(wc -l < results/god_v3.jsonl) rows, errors $(grep -c '"error"' results/god_v3.jsonl || true)"
done
echo "$(date '+%F %T') retry"
"${J[@]}" retry --workers 16 | tail -3
echo "$(date '+%F %T') DONE"
