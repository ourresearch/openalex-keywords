# openalex-keywords

This repository contains the code for generating OpenAlex keywords. Each model iteration has its own directory with a full explanation. To learn more, visit https://openalex.org/

### Model Iterations
* [v3](v3/) (current, shipped 2026-09-29): Claude Opus 5.5 labels about 1M works as an indexer; a fine-tuned Qwen3-4B tags all 474M works; a rule-based vocabulary build and synonym merge produce 1,874,460 keywords. Code, prompts, gold sets, evaluations and data are in [v3/](v3/).
* [v2](v2/) (superseded by v3): 65,004 keywords derived from each work's topics, matched with bge-m3.
* [v1](v1/) (superseded)
