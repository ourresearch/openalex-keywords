"""Synonym-merge candidates (#1322, peer review 2026-09-26): embed every v3 vocabulary keyword (bge-m3, fp16) and find, for each keyword, its nearest
neighbours by cosine on the GPU (chunked matmul, no faiss). Input /vol/synonyms/vocab.jsonl {kid, display_name, works_count}; output
/vol/synonyms/pairs.jsonl {a, b, cos} for cos >= MIN_COS, a != b, both directions possible (dedupe downstream). modal run modal_embed_knn.py"""
import modal, json, time
app = modal.App("oxjob1322-synonyms")
vol = modal.Volume.from_name("oxjob1322")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0", "sentence-transformers>=3.0", "numpy").env({"HF_HOME": "/vol/hf"})
@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=7200, memory=131072, cpu=16)
def run(vocab: str = "/vol/synonyms/vocab.jsonl", out: str = "/vol/synonyms/pairs.jsonl", min_cos: float = 0.88, k: int = 8):
    MIN_COS, K = min_cos, k
    import torch, numpy as np
    from sentence_transformers import SentenceTransformer
    rows = [json.loads(l) for l in open(vocab)]; names = [r["display_name"] for r in rows]; t0 = time.time()
    m = SentenceTransformer("BAAI/bge-m3", device="cuda"); m.half()
    E = m.encode(names, batch_size=1024, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False).half()
    print(f"embedded {len(names):,} in {time.time()-t0:.0f}s", flush=True)
    out = open(out, "w"); n = 0
    for i in range(0, len(names), 4096):
        S = E[i:i + 4096] @ E.T                                     # (chunk, N)
        v, idx = torch.topk(S, K + 1, dim=1)
        v, idx = v.float().cpu().numpy(), idx.cpu().numpy()
        for r in range(v.shape[0]):
            a = i + r
            for c, b in zip(v[r], idx[r]):
                if b == a or c < MIN_COS: continue   # both directions may be written; dedupe downstream
                out.write(json.dumps({"a": rows[a]["kid"], "b": rows[b]["kid"], "cos": round(float(c), 4)}) + "\n"); n += 1
    out.close(); vol.commit(); print(f"{n:,} pairs >= {MIN_COS} in {time.time()-t0:.0f}s", flush=True); return n
@app.local_entrypoint()
def main(vocab: str = "/vol/synonyms/vocab.jsonl", out: str = "/vol/synonyms/pairs.jsonl", min_cos: float = 0.88, k: int = 8): print(run.remote(vocab, out, min_cos, k))
