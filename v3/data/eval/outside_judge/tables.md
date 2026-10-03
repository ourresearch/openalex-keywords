## Measurement A: per-keyword accuracy

| Slice | List | Works | Keywords | Accurate (Wilson 95% CI) | Work-cluster bootstrap CI | Per-work mean accuracy | Wrong/unsupported/wrong-sense | Too generic | Not aboutness | Malformed | Redundant (flag) |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| All | old | 400 | 2708 | 43.5% (41.7–45.4) | 41.1–45.8 | 33.9% | 41.0% | 14.3% | 1.0% | 0.1% | 2.2% |
| All | new | 400 | 2357 | 91.1% (89.9–92.2) | 89.6–92.5 | 89.3% | 8.4% | 0.1% | 0.4% | 0.0% | 1.9% |
| With abstract | old | 282 | 2211 | 46.9% (44.8–49.0) | 44.3–49.3 | 40.2% | 37.3% | 14.5% | 1.2% | 0.1% | 2.5% |
| With abstract | new | 282 | 1866 | 94.4% (93.3–95.4) | 93.1–95.7 | 94.5% | 5.4% | 0.1% | 0.2% | 0.0% | 1.6% |
| No abstract | old | 118 | 497 | 28.6% (24.8–32.7) | 23.3–33.5 | 18.7% | 57.3% | 13.7% | 0.4% | 0.0% | 0.8% |
| No abstract | new | 118 | 491 | 78.4% (74.6–81.8) | 73.7–82.9 | 77.0% | 20.0% | 0.2% | 1.4% | 0.0% | 3.1% |
| Non-xpac | old | 291 | 2184 | 44.9% (42.8–47.0) | 42.3–47.3 | 34.5% | 37.2% | 17.1% | 0.8% | 0.0% | 2.0% |
| Non-xpac | new | 291 | 1803 | 92.2% (90.8–93.3) | 90.5–93.7 | 89.5% | 7.5% | 0.1% | 0.2% | 0.0% | 1.8% |
| xpac | old | 109 | 524 | 38.0% (33.9–42.2) | 31.7–43.9 | 32.2% | 56.9% | 2.9% | 1.9% | 0.4% | 2.9% |
| xpac | new | 109 | 554 | 87.5% (84.5–90.0) | 83.8–90.9 | 88.8% | 11.4% | 0.0% | 1.1% | 0.0% | 2.2% |

Per-work accuracy, new vs old (from the single-list judgments): {'new higher': 385, 'equal': 10, 'old higher': 5}

## Preference (head-to-head)

| Slice | n | New preferred | Old preferred | Tie | Neither useful |
|---|---:|---:|---:|---:|---:|
| All | 400 | 395 | 1 | 0 | 4 |
| With abstract | 282 | 282 | 0 | 0 | 0 |
| No abstract | 118 | 113 | 1 | 0 | 4 |
| Non-xpac | 291 | 286 | 1 | 0 | 4 |
| xpac | 109 | 109 | 0 | 0 | 0 |

Raw A/B picks (position-bias check): {'B': 189, 'A': 207, 'neither_useful': 4}

## Measurement B: antimicrobial resistance

| Stratum | n | GPT yes (95% CI) | GPT yes+partly (95% CI) | Claude Opus 5.5 yes | Claude yes+partly |
|---|---:|---|---|---|---|
| kw_only | 150 | 64.0% (56.1–71.2) | 76.7% (69.3–82.7) | 61% | 87% |
| kw_only_noabs | 100 | 81.0% (72.2–87.5) | 90.0% (82.6–94.5) | 69% | 96% |
| text | 50 | 80.0% (67.0–88.8) | 88.0% (76.2–94.4) | 68% | 96% |

Claude (rows) vs GPT (cols) verdicts, unique works:
  claude yes   : gpt yes=191  gpt partly=2  gpt no=0
  claude partly: gpt yes=24  gpt partly=30  gpt no=26
  claude no    : gpt yes=1  gpt partly=0  gpt no=25

