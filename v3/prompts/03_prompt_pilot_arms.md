# Prompt pilot arms (2026-09-25)

- **Used for:** choosing the teacher prompt on 620 fixed works (docs/history.md, § prompt pilot). Arm C = arm B's text on the lean input (title, abstract, venue); arms D, E, F use the lean input; A and B the v2 input with Year / Type / Language.
- **Source:** `code/teacher/pilot2/run_arms.py`
- **Model:** `claude-opus-5-5`, effort `medium`, `max_tokens` 4000, server-side fallback beta as for the teacher.
- **Output schema:** {"keywords": [string]} (strict JSON schema)

**Arm A**

```text
Assign keywords that describe what this work is about.
```

**Arm B**

```text
Assign keywords that describe what this work is about. Describe its subject matter: not the kind of document it is, and not a country or place unless the work is about that place. Write the keywords in English.
```

**Arm D**

```text
Assign keywords that describe what this work is about. Describe its subject matter: not the kind of document it is, and not a country or place unless the work is about that place. Write the keywords in English. If the record says nothing about its subject, return an empty list.
```

**Arm E**

```text
Assign keywords that describe what this work is about. Write the keywords in English. If the record says nothing about its subject, return an empty list.
```

**Arm F** (shipped as v3)

```text
Assign keywords that describe what this work is about. Describe its subject matter: not the kind of document or record it is, not its general field, not standard methods it merely uses, and not a country or place unless the work is about that place. Write the keywords in English. If the record says nothing about its subject, return an empty list.
```
