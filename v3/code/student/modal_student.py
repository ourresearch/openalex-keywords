"""Generative student on Modal (H100/H200). Data + scripts live on Modal Volume `oxjob1322` (/vol): data/student_train_all.jsonl, data/student_eval.jsonl, student_gen_train.py, student_gen_infer.py.
Launch (detached, returns immediately):  modal run --detach modal_student.py --tag qwen17all --base Qwen/Qwen3-1.7B --n 940000 --epochs 1 --bs 16 --acc 1 --gpu H100
Outputs on the Volume: /vol/models/student_<tag>, /vol/results/train_<tag>.log, /vol/results/arm_<tag>.jsonl (fetch with `modal volume get oxjob1322 results/arm_<tag>.jsonl`)."""
import modal, subprocess, os, shutil, time, sys
app = modal.App("oxjob1322-student")
vol = modal.Volume.from_name("oxjob1322", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0", "transformers>=4.51,<5.0", "accelerate>=1.0", "bitsandbytes>=0.45", "huggingface_hub", "numpy")
         .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"}))
def _run(cmd, log, launcher=None):
    print("==", " ".join(cmd), flush=True); t0 = time.time()
    p = subprocess.Popen((launcher or [sys.executable]) + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd="/tmp/work", text=True)
    with open(log, "w") as f:
        for line in p.stdout:
            f.write(line)
            if line.startswith(("step", "saved", "cuda", "Traceback", "torch.", "RuntimeError")) or "procs" in line or "rows" in line: print(line.rstrip()[:200], flush=True)
    r = p.wait(); print(f"   exit {r} {time.time()-t0:.0f}s", flush=True); vol.commit()
    if r: print(open(log).read()[-3000:], flush=True); raise RuntimeError(f"failed: {cmd[0]}")
def _train_and_infer(tag, base, n, epochs, bs, acc, maxlen, optim8bit, train_file, ngpu, gradckpt=False, eval_file="student_eval.jsonl"):
    os.makedirs("/tmp/work", exist_ok=True); os.makedirs("/vol/results", exist_ok=True); os.makedirs("/vol/models", exist_ok=True)
    for fn in ("student_gen_train.py", "student_gen_train_ddp.py", "student_gen_infer.py"): shutil.copy(f"/vol/{fn}", f"/tmp/work/{fn}")
    m = f"/vol/models/student_{tag}"
    if not os.path.exists(f"{m}/config.json"):
        common = ["--train", f"/vol/data/{train_file}", "--out", m, "--limit", str(n), "--epochs", str(epochs), "--bs", str(bs), "--grad-accum", str(acc), "--maxlen", str(maxlen), "--base", base] + (["--optim8bit"] if optim8bit else [])
        if ngpu > 1: _run([f"/tmp/work/student_gen_train_ddp.py"] + common + ([] if gradckpt else ["--no-grad-ckpt"]), f"/vol/results/train_{tag}.log", launcher=[sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu", "--num_processes", str(ngpu), "--mixed_precision", "no"])
        else: _run([f"/tmp/work/student_gen_train.py"] + common, f"/vol/results/train_{tag}.log")
    _run([f"/tmp/work/student_gen_infer.py", "--model", m, "--tag", tag, "--eval", f"/vol/data/{eval_file}", "--bs", "64"], f"/vol/results/infer_{tag}.log")
    return open(f"/vol/results/infer_{tag}.log").read()[-300:]
@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=43200, memory=65536, cpu=8)
def train_and_infer(tag: str, base: str, n: int, epochs: float, bs: int, acc: int, maxlen: int, optim8bit: bool, train_file: str):
    return _train_and_infer(tag, base, n, epochs, bs, acc, maxlen, optim8bit, train_file, 1)
@app.function(image=image, gpu="H200:8", volumes={"/vol": vol}, timeout=21600, memory=524288, cpu=32)   # H200: the 4B at bs 8 OOMs on an 80 GB H100 (2026-09-26); earlier "H100" runs were served as H200s
def train_and_infer_8(tag: str, base: str, n: int, epochs: float, bs: int, acc: int, maxlen: int, optim8bit: bool, train_file: str, gradckpt: bool = False, eval_file: str = "student_eval.jsonl"):
    return _train_and_infer(tag, base, n, epochs, bs, acc, maxlen, optim8bit, train_file, 8, gradckpt, eval_file)
@app.local_entrypoint()
def main(tag: str, base: str = "Qwen/Qwen3-1.7B", n: int = 940000, epochs: float = 1.0, bs: int = 16, acc: int = 1, maxlen: int = 640, optim8bit: bool = False, train_file: str = "student_train_all.jsonl", gpus: int = 1, gradckpt: bool = False, eval_file: str = "student_eval.jsonl"):
    if gpus == 8: print(train_and_infer_8.remote(tag, base, n, epochs, bs, acc, maxlen, optim8bit, train_file, gradckpt, eval_file))
    else: print(train_and_infer.remote(tag, base, n, epochs, bs, acc, maxlen, optim8bit, train_file))
