"""Keyword id = keywords/<kid>. Two layers:
kid0(s) = slug(canon(s)), a pure string function (this file; the Databricks UDF openalex.transient.kw1322_kid0 is created from it):
  canon = NFKC, lower, hyphen/underscore/slash -> space, collapse spaces, strip edge punctuation (no plural folding here);
  slug = openalex-text-api keywords.keyword_slug (Greek letters spelled out, NFKD ASCII fold, lower, non-alphanumeric runs -> '-'),
  except that a string with letters outside Latin/Greek keeps its Unicode letters (ASCII folding would erase it).
kid = plural merge on top of kid0, data-driven, built in SQL (build_vocab.py step 'fold', table kw1322_fold): a kid0 ending in s merges with its singular
  (-ies -> -y, -(ss|sh|ch|x|z)es -> -\\1, -s -> '') only when that singular exists with >= 5% of the plural's mentions; the group's id is its most common member.
  (A blind "strip the final s" rule made apoptosi, diabetes-mellitu, pari, united-state; 2026-09-25.)"""
import re, unicodedata
GREEK = str.maketrans({"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu", "π": "pi", "σ": "sigma", "τ": "tau", "ω": "omega",
                       "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "Ω": "omega"})
def canon(s):
    s = unicodedata.normalize('NFKC', s).lower().strip(); s = re.sub(r'[‐-―_/\-]', ' ', s); s = re.sub(r'\s+', ' ', s)
    return s.strip(' .,;:"\'')
def kid0(s):
    if s is None: return None
    c = canon(s)
    if any(ch.isalpha() and not unicodedata.name(ch, '').startswith(('LATIN', 'GREEK')) for ch in c):
        return re.sub(r'[\W_]+', '-', c).strip('-') or None
    a = unicodedata.normalize('NFKD', c.translate(GREEK)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', a).strip('-') or None
