"""v3 corpus input (#1322, 2026-09-26): every work in openalex_works (xpac included), the fields the v3 student sees (title, abstract ≤ 6000 chars, venue),
sharded by pmod(hash(id), 4000) (≈ 119K works per shard) and clustered by shard so each worker's read is pruned. uv run --with httpx python build_input.py"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbx
t0 = time.time()
dbx.sql("""CREATE OR REPLACE TABLE openalex.transient.kw_corpus_input_v3 CLUSTER BY (shard) AS
SELECT pmod(hash(w.id), 4000) AS shard, w.id, w.title, substr(w.abstract, 1, 6000) AS abstract, w.primary_location.source.display_name AS journal, COALESCE(w.is_xpac, false) AS is_xpac
FROM openalex.works.openalex_works w""", warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'), timeout_s=4 * 3600, verbose=False)
print(f'built in {time.time() - t0:.0f}s')
print(dbx.sql("SELECT count(*) n, count_if(is_xpac) xpac, count(DISTINCT shard) shards, max(c) max_shard FROM (SELECT *, count(*) OVER (PARTITION BY shard) c FROM openalex.transient.kw_corpus_input_v3)", warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'), verbose=False))
