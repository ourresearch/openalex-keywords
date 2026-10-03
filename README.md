# openalex-keywords

This repository contains the code for generating OpenAlex keywords. Each model iteration has its own directory with a full explanation. To learn more, visit https://openalex.org/

### Model Iterations
* [v3](v3/) (current, shipped 2026-09-29; release v3.1 on 2026-10-03): Claude Opus 5.5 labels about 1M works as an indexer; a fine-tuned Qwen3-4B tags all 474M works; a rule-based vocabulary build and synonym merge produce 1.88 million keywords on 419.6M works. An outside judge (OpenAI GPT-6 Astra) rates 91% of them accurate, against 44% for v2. v3.1 adds the synonym audit, keyword sense splits, alternative names for search, and the outside-judge evaluation. Code, prompts, gold sets, evaluations and data are in [v3/](v3/).
* [v2](v2/) (superseded by v3): 65,004 keywords derived from each work's topics, matched with bge-m3.
* [v1](v1/) (superseded)
