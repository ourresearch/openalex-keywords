"""Teacher prompt v3 (Jason 2026-09-25, #1322 EXPLORE § 23: pilot2 arm F). Minimal aboutness instruction, no count, may abstain; input = title + abstract + venue only
(Type / Language / Year removed: they caused document-type and country keywords). Same interface as prompt.py."""
SYSTEM = ("Assign keywords that describe what this work is about. Describe its subject matter: not the kind of document or record it is, not its general field, not standard methods it merely uses, "
          "and not a country or place unless the work is about that place. Write the keywords in English. If the record says nothing about its subject, return an empty list.")
SCHEMA = {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}}, "required": ["keywords"], "additionalProperties": False}
def user_text(w):
    return "\n".join([f"Title: {w.get('title') or '(none)'}", f"Abstract: {(w.get('abstract') or '(none)')[:6000]}", f"Venue: {w.get('journal') or w.get('venue') or '(unknown)'}"])
