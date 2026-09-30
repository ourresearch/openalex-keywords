# Student weights

The model that assigns OpenAlex keywords: a full fine-tune of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) (bf16, 7.5 GiB), trained on Claude Opus 5.5 labels (prompt: [`prompts/01_teacher_v3.md`](../prompts/01_teacher_v3.md)). The weights are assets of the GitHub [`v3.0` release](https://github.com/ourresearch/openalex-keywords/releases/tag/v3.0), split into five safetensors files under 2 GB (GitHub's per-file limit) with a standard `model.safetensors.index.json`, so Transformers and vLLM load the folder as is. The split is lossless: every tensor is byte-identical to the trained checkpoint. License: Apache-2.0, like the base model.

## Download

```bash
mkdir keywords-v3-qwen3-4b && cd keywords-v3-qwen3-4b
gh release download v3.0 -R ourresearch/openalex-keywords -p 'model-*' -p '*.json' -p 'merges.txt' -p 'chat_template.jinja'
```

## Input format

Plain text, no chat template, no special tokens:

```
Title: {title or "(none)"}
Abstract: {first 6000 characters of the abstract, or "(none)"}
Venue: {journal or source name, or "(unknown)"}
Keywords:
```

The model continues with ` keyword one; keyword two; ...` and an end-of-sequence token, best first, and may return nothing when the record says nothing about its subject. OpenAlex truncates the text to 4,000 characters and the prompt to 560 tokens, decodes greedily, and scores each keyword by exp(mean token log-probability) ([`code/tagging/modal_corpus_v3.py`](../code/tagging/modal_corpus_v3.py)). Raw keywords then pass through the vocabulary and synonym rules ([`vocabulary.md`](vocabulary.md), [`synonyms.md`](synonyms.md)) before they become OpenAlex keyword ids.

```python
from vllm import LLM, SamplingParams
llm = LLM("./keywords-v3-qwen3-4b")
text = "Title: Observation of Gravitational Waves from a Binary Black Hole Merger\nAbstract: ...\nVenue: Physical Review Letters"
out = llm.generate(text[:4000] + "\nKeywords:", SamplingParams(temperature=0, max_tokens=160))
print([k.strip() for k in out[0].outputs[0].text.split(";") if k.strip()])
```
