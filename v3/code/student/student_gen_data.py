"""Build the generative student's training / eval files from the God results: data/student_train.jsonl, data/student_eval.jsonl rows {id, prompt, target}.
prompt = the same work text the indexer saw (title / abstract / venue / year / type / language); target = Opus's keywords joined by '; ' in rank order.
python student_gen_data.py [--tags god55,god55w2,...] [--limit-train N]
v3 relabel: python student_gen_data.py --tags v3 --prompt prompt_v3 --samples relabel/data/sample.jsonl --eval-ids relabel/data/eval_ids.txt --suffix _v3 (prompt = title / abstract / venue)"""
import argparse, json, sys, os
sys.path.insert(0,'.'); import importlib
ap=argparse.ArgumentParser(); ap.add_argument('--tags', default='god55'); ap.add_argument('--limit-train', type=int); ap.add_argument('--prompt', default='prompt'); ap.add_argument('--eval-ids', default='data/god_eval_ids.txt'); ap.add_argument('--suffix', default=''); ap.add_argument('--samples', default='data/god_sample.jsonl,data/god_sample_w2.jsonl,data/god_sample_w3.jsonl,data/god_sample_w4.jsonl,data/god_sample_w5.jsonl,data/god_sample_w6.jsonl,data/god_sample_w7.jsonl'); a=ap.parse_args(); user_text=importlib.import_module(a.prompt).user_text
W={}
for p in a.samples.split(','):
    if os.path.exists(p):
        for l in open(p): w=json.loads(l); W[w['id']]=w
ev=set(open(a.eval_ids).read().split()); tr=[]; te=[]
for t in a.tags.split(','):
    for r in map(json.loads, open(f'results/god_{t}.jsonl')):
        if 'answer' not in r or r['id'] not in W: continue
        row={'id':r['id'],'prompt':user_text(W[r['id']]),'target':'; '.join(k.strip() for k in r['answer']['keywords'] if k.strip())}
        (te if r['id'] in ev else tr).append(row)
if a.limit_train: tr=tr[:a.limit_train]
for name,rows in (('train',tr),('eval',te)):
    with open(f'data/student_{name}{a.suffix}.jsonl','w') as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False)+'\n')
    print(name, len(rows))
