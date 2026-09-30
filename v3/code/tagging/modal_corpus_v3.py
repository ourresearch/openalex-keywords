"""v3 corpus retag on Modal (#1322, 2026-09-26; Jason: parallelise to the workspace GPU cap). One call = one shard of openalex.transient.kw_corpus_input_v3
(pmod(hash(id), 4000), ≈ 119K works): the worker runs its own Databricks statement (no presigned links waiting in a queue to expire), tags with vLLM + the v3
student, keeps a per-keyword confidence (mean token log-prob of the keyword's tokens), and PUTs a gzipped jsonl straight to the Databricks Volume
(no Modal Volume hop, nothing on desk). Rows: {id, keywords:[...], lp:[...], model, shard}. Idempotent: the file name is the shard, overwrite=true.
Deploy:  modal deploy modal_corpus_v3.py    Driver: corpus_run_v3.py.  GPU / quantisation per run: Tagger.with_options(gpu="H100")(model=..., quant="fp8")."""
import modal, json, time, os
app = modal.App("oxjob1322-corpus-v3")
vol = modal.Volume.from_name("oxjob1322")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11").pip_install("vllm>=0.10", "transformers>=4.51,<5.0", "httpx")
         .env({"HF_HOME": "/vol/hf", "CUDA_HOME": "/usr/local/cuda", "VLLM_USE_FLASHINFER_SAMPLER": "0"}))
DBX_VOL = "/Volumes/openalex/authors/oxjob_tmp/oxjob1322_god/corpus_v3"
TABLE_IN = "openalex.transient.kw_corpus_input_v3"
WAREHOUSE = os.environ.get('DATABRICKS_WAREHOUSE_ID')   # DEV-WH (X-Large serverless): the shared Medium warehouse queued 49 concurrent shard reads into 900 s timeouts (2026-09-27 20:45 CT)
def user_text(w):   # = god/prompt_v3.user_text (the student's training prompt)
    return "\n".join([f"Title: {w.get('title') or '(none)'}", f"Abstract: {(w.get('abstract') or '(none)')[:6000]}", f"Venue: {w.get('journal') or '(unknown)'}"])
def keyword_lps(tok, token_ids, logprobs, special=frozenset()):
    """Split the generated text on ';' and average the sampled-token log-probs per keyword (separator tokens count toward the keyword they close).
    Special tokens (the <|im_end|> that ends every list) are skipped: vLLM keeps the stop token in token_ids (bug in the first 330 shards, 2026-09-26)."""
    kws, lps, cur, cur_lp = [], [], "", []
    for tid, d in zip(token_ids, logprobs):
        if tid in special: continue
        piece = tok.decode([tid]); lp = d[tid].logprob if d and tid in d else 0.0
        parts = piece.split(";")
        cur += parts[0]; cur_lp.append(lp)
        for p in parts[1:]:
            if cur.strip(): kws.append(cur.strip()); lps.append(round(sum(cur_lp) / len(cur_lp), 4))
            cur, cur_lp = p, []
    if cur.strip() and cur_lp: kws.append(cur.strip()); lps.append(round(sum(cur_lp) / len(cur_lp), 4))
    return kws, lps
@app.cls(image=image, gpu="B200", volumes={"/vol": vol}, secrets=[modal.Secret.from_name("oxjob-databricks")], timeout=900, memory=65536, cpu=8,
         max_containers=10, scaledown_window=120, retries=3)   # Jason 2026-09-27 22:32: max 10 GPUs so 40 stay free for affiliations / authors work
class Tagger:
    model: str = modal.parameter(default="/vol/models/student_qwen4b_v3")
    quant: str = modal.parameter(default="")
    out_dir: str = modal.parameter(default=DBX_VOL)   # Databricks Volume folder for this run's shard files
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        kw = dict(model=self.model, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.9, enable_prefix_caching=False,
                  max_num_seqs=512, max_num_batched_tokens=32768)   # 4B on a 180 GB B200: 256 left decode under-batched; 1024 coincided with hung workers (2026-09-26)
        if self.quant: kw["quantization"] = self.quant
        self.llm = LLM(**kw); self.tok = AutoTokenizer.from_pretrained(self.model); self.special = frozenset(self.tok.all_special_ids)
        self.sp = SamplingParams(temperature=0.0, max_tokens=192, stop=["\n"], logprobs=1)
    def _tag(self, texts, chunk=16384):
        """Chunks keep the GIL-bound tokenise / detokenise steps short so Modal's heartbeat gets through (one 118K-prompt call starved it, 2026-09-26)."""
        out = []
        for i in range(0, len(texts), chunk):
            prompts = [{"prompt_token_ids": self.tok(t + "\nKeywords:", add_special_tokens=False)["input_ids"][:560]} for t in texts[i:i + chunk]]
            out += [keyword_lps(self.tok, o.outputs[0].token_ids, o.outputs[0].logprobs, self.special) for o in self.llm.generate(prompts, self.sp, use_tqdm=False)]
        return out
    @modal.method()
    def tag_texts(self, texts: list):
        """Eval / throughput probe: returns (outputs, seconds)."""
        t0 = time.time(); out = self._tag(texts); return out, round(time.time() - t0, 2)
    @modal.method()
    def run_shard(self, shard: int):
        import httpx, gzip
        t0 = time.time(); host = os.environ["DATABRICKS_HOST"].rstrip("/"); H = {"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"}
        with httpx.Client(timeout=300, headers=H) as h:
            st = h.post(f"{host}/api/2.0/sql/statements", json={"warehouse_id": WAREHOUSE, "wait_timeout": "30s", "disposition": "EXTERNAL_LINKS", "format": "JSON_ARRAY",
                        "statement": f"SELECT id, title, abstract, journal FROM {TABLE_IN} WHERE shard = {shard}"}).json()
            t_q = time.time()
            while st["status"]["state"] in ("PENDING", "RUNNING"):
                if time.time() - t_q > 300:   # a queued / stuck read: cancel it server-side and fail fast so Modal retries (no silent 900 s timeouts)
                    h.post(f"{host}/api/2.0/sql/statements/{st['statement_id']}/cancel"); raise TimeoutError(f"shard {shard}: statement not done in 300 s")
                time.sleep(3); st = h.get(f"{host}/api/2.0/sql/statements/{st['statement_id']}").json()
            if st["status"]["state"] != "SUCCEEDED": raise RuntimeError(f"shard {shard}: {json.dumps(st['status'])[:400]}")
            cols = [c["name"] for c in st["manifest"]["schema"]["columns"]]; rows = []
            for i in range(st["manifest"].get("total_chunk_count", 0)):
                ch = h.get(f"{host}/api/2.0/sql/statements/{st['statement_id']}/result/chunks/{i}").json()
                for l in ch.get("external_links", []): rows += [dict(zip(cols, r)) for r in httpx.get(l["external_link"], timeout=300).json()]
        t_fetch = time.time() - t0
        out = self._tag([user_text(w) for w in rows]); t_gen = time.time() - t0 - t_fetch
        model = os.path.basename(self.model) + (f"+{self.quant}" if self.quant else "")
        body = gzip.compress("".join(json.dumps({"id": f"W{w['id']}", "keywords": k, "lp": lp, "model": model, "shard": shard}, ensure_ascii=False) + "\n"
                                     for w, (k, lp) in zip(rows, out)).encode(), 6)
        r = httpx.put(f"{host}/api/2.0/fs/files{self.out_dir}/{shard:04d}.jsonl.gz?overwrite=true", headers={**H, "Content-Type": "application/octet-stream"}, content=body, timeout=600)
        if r.status_code >= 300: raise RuntimeError(f"shard {shard}: PUT {r.status_code} {r.text[:300]}")
        return {"shard": shard, "works": len(rows), "fetch_s": round(t_fetch, 1), "gen_s": round(t_gen, 1), "total_s": round(time.time() - t0, 1), "bytes": len(body)}
