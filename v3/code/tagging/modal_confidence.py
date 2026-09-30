"""Confidence cut, measurement (#1322 Next Action 2): on the 10K eval, (a) regenerate greedy with logprobs (top-5 per output token) and (b) re-score the same outputs prefill-only (prompt_logprobs over prompt + output, max_tokens=1); time both on one H100.
Writes /vol/results/conf_eval.jsonl: {id, tokens:[{t, lp, top:{tok: lp}}], text, prefill_lp:[lp per output token]} for local analysis (conf_analyze.py).
modal run modal_confidence.py [--n 10000]"""
import modal, json, time, os
app = modal.App("oxjob1322-confidence")
vol = modal.Volume.from_name("oxjob1322")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11").pip_install("vllm>=0.8", "transformers>=4.51,<5.0")
         .env({"HF_HOME": "/vol/hf", "CUDA_HOME": "/usr/local/cuda", "VLLM_USE_FLASHINFER_SAMPLER": "0"}))
MODEL = "/vol/models/student_qwen4ball8"
@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=7200, memory=65536, cpu=8)
def run(n: int):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in open("/vol/data/student_eval.jsonl")][:n]
    tok = AutoTokenizer.from_pretrained(MODEL)
    P = [tok(r["prompt"][:4000] + "\nKeywords:", add_special_tokens=False)["input_ids"][:560] for r in rows]
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.9, enable_prefix_caching=False)
    # warm-up so neither timing carries compile/graph capture
    llm.generate([{"prompt_token_ids": p} for p in P[:256]], SamplingParams(temperature=0.0, max_tokens=96, stop=["\n"]))
    t0 = time.time(); plain = llm.generate([{"prompt_token_ids": p} for p in P], SamplingParams(temperature=0.0, max_tokens=96, stop=["\n"])); t_plain = time.time() - t0
    t0 = time.time(); gen = llm.generate([{"prompt_token_ids": p} for p in P], SamplingParams(temperature=0.0, max_tokens=96, stop=["\n"], logprobs=5)); t_gen = time.time() - t0
    O = [list(g.outputs[0].token_ids) for g in gen]
    t0 = time.time(); pre = llm.generate([{"prompt_token_ids": p + o} for p, o in zip(P, O)], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)); t_pre = time.time() - t0
    same = sum(a.outputs[0].text == b.outputs[0].text for a, b in zip(plain, gen))
    with open("/vol/results/conf_eval.jsonl", "w") as f:
        for r, p, g, q in zip(rows, P, gen, pre):
            o = g.outputs[0]; toks = []
            for tid, lpd in zip(o.token_ids, o.logprobs):
                toks.append({"t": tok.decode([tid]), "lp": round(lpd[tid].logprob, 4), "top": {tok.decode([k]): round(v.logprob, 4) for k, v in lpd.items()}})
            pl = [round(d[t].logprob, 4) for t, d in zip(o.token_ids, q.prompt_logprobs[len(p):])] if q.prompt_logprobs else []
            f.write(json.dumps({"id": r["id"], "text": o.text, "finish": o.finish_reason, "tokens": toks, "prefill_lp": pl}, ensure_ascii=False) + "\n")
    vol.commit()
    msg = f"{len(rows)} works: plain generate {t_plain:.1f}s ({len(rows)/t_plain:.0f}/s); generate+logprobs {t_gen:.1f}s ({len(rows)/t_gen:.0f}/s); prefill re-score {t_pre:.1f}s ({len(rows)/t_pre:.0f}/s); identical text plain vs logprobs {same}/{len(rows)}"
    print(msg, flush=True); return msg
@app.local_entrypoint()
def main(n: int = 10000):
    print(run.remote(n))
