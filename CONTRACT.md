# Team contract (agree on this before writing code)

## Work split (4 people)
| # | Role | Files |
|---|------|-------|
| 1 | Ingestion & Retrieval | src/ingest.py, src/retriever.py, scripts/reset.py, notebooks/01_eda.ipynb, 02_retrieval.ipynb |
| 2 | Graph, Router & Memory | src/graph.py, src/router.py, src/memory.py, src/prompts.py |
| 3 | Tools, UI & Logging | src/tools.py, app.py, src/logging_setup.py |
| 4 | Evaluation, Docs & Config | tests/evaluate.py, src/settings.py, requirements.txt, README, docs/, outputs/final_report.md, notebooks/03 |

## Shared rules
- Shared types live in src/schemas.py. Do not change them without telling everyone.
- Shared values live in src/settings.py. No hard-coded numbers elsewhere.
- Domains are exactly: "hr", "it", "finance". Languages: "en", "ar", "bilingual".
- Interfaces (signatures) are stubs in each file. Keep them as written.
- Run `black .` and `ruff check .` before every commit.
- Branch per person, merge via pull request.

## To decide together
- LLM, embeddings model, vector store (write the choice in settings.py)
- Pin versions in requirements.txt
