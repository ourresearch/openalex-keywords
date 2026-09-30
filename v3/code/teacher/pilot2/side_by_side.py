"""20 pilot works side by side for Jason to read and pick an arm -> SIDE_BY_SIDE.md (committed). Seeded pick: 5 abstract, 5 title-only, 3 non-English, 7 xpac.
Marks: ~~struck~~ = the judge rejected it; (doc-type) / (country, no evidence) = mechanical flags from the vocab-build lists."""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]
import score   # loads arms, judge verdicts, fate()
from kid import kid0
NAMES = {'v2': 'current prompt', 'A': 'minimal', 'B': 'minimal + aboutness', 'C': 'aboutness, lean input', 'D': 'aboutness, lean input, may abstain', 'E': 'minimal, English, lean input, may abstain'}
rng = random.Random(7); out = ['# Prompt pilot: 20 works side by side (#1322)', '',
    'Each work shows the record, then each prompt\'s keywords in its own order. ~~Struck~~ = the Opus judge rejected it (strict rubric: supported by the record, about the work, not its document type). '
    'Flags: *(doc-type)*, *(country, no evidence)* = what the vocabulary rules would purge. Prompts: **current** = the God-label prompt (3–8 keywords, sees Type / Language / Year); '
    '**minimal** = "Assign keywords that describe what this work is about."; **+ aboutness** adds "Describe its subject matter: not the kind of document it is, and not a country or place unless the work is about that place. Write the keywords in English."; '
    '**lean input** = the same prompt shown only title, abstract and venue; **may abstain** adds "If the record says nothing about its subject, return an empty list."; **minimal, English, lean input, may abstain** = the last one without the aboutness sentence (Jason\'s question: are the prohibitions still needed?)', '']
for g, n in [('abstract', 5), ('title_only', 5), ('non_english', 3), ('xpac', 7)]:
    ids = sorted(i for i in score.common if score.works[i]['group'] == g); rng.shuffle(ids)
    for i in ids[:n]:
        w = score.works[i]; ab = (w.get('abstract') or '').strip(); v = score.judged.get(i, {}).get('verdict', {})
        out += [f'## {w.get("title") or "(no title)"}', f'`{i}` · {g.replace("_", " ")} · {w.get("type")} · {w.get("language")} · {w.get("journal") or "no venue"}' + (f' · judge: record insufficient' if score.judged.get(i, {}).get('sufficient') is False else ''), '',
                f'> {ab[:500] + ("…" if len(ab) > 500 else "")}' if ab else '> (no abstract)', '']
        for arm in score.ARMS:
            parts = []
            for k in score.A[arm][i]:
                f = score.fate(k, w); s = k if v.get(kid0(k), True) else f'~~{k}~~'
                if f == 'purge' or f == 'dual_redundant': s += ' *(doc-type)*'
                if f == 'country_no_evidence': s += ' *(country, no evidence)*'
                parts.append(s)
            out.append(f'- **{NAMES[arm]}** ({len(parts)}): ' + ('; '.join(parts) or '*(empty)*'))
        out.append('')
open(f'{HERE}/SIDE_BY_SIDE.md', 'w').write('\n'.join(out) + '\n'); print('wrote SIDE_BY_SIDE.md')
