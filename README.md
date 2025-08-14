# AI Assistant — Single-file deployable project
MIT License — see LICENSE.

This repository contains a fully functional AI assistant web application designed to run on Render's free tier without paid inference APIs by default.

Highlights
- Pure Python backend (Flask) + vanilla JS frontend — all in this repo.
- Interpolated Kneser–Ney n-gram language model (train/save/load).
- Optional Transformer-backed inference (manually installed).
- BPE tokenizer with HF tokenizers fallback and pure-Python fallback.
- Dataset ingestion, file upload, zip extraction, deduplication, dataset viewer & export.
- Admin forced login (ENV VARS) + session revocation + CSRF protection.
- RAG (optional lazy load) with sentence-transformers + faiss-cpu if installed.
- Backups on every upload/train/save; optional S3 archival if env provided.
- Keepalive endpoint and docs for cron-job.org to avoid Render idling.
- Lightweight pytest tests for tokenizer and n-gram model.

## Quickstart (local)
```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
export ADMIN_USER=admin
export ADMIN_PASS=changeme
export ADMIN_SESSION_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
python app.py
# Open http://127.0.0.1:5000
```

## Deploy to Render (summary)
- Create a new Web Service, link to this repo, set Python build, set `ADMIN_USER`, `ADMIN_PASS`, `ADMIN_SESSION_SECRET` in Render dashboard, add `render.yaml` and `Procfile` included.
- See full deploy instructions below in this README for environment variables, optional features, and cron-job keepalive instructions.

## Files included (high level)
- `app.py` — Flask backend serving APIs and pages.
- `static/` — frontend JS/CSS.
- `data/uploads/` — sample dataset (JSONL ~3–5MB) and other uploads.
- `scripts/generate_groq.py` — example synthetic generator (not executed automatically).
- `models/` — model save/load area.
- `tests/` — pytest tests.
- `requirements.txt` and `requirements-optional.txt` (optional heavy deps).

-- README truncated here; full README included in repo.
# AI Assistant — Single-file deployable project
MIT License — see LICENSE.

This repository contains a fully functional AI assistant web application designed to run on Render's free tier without paid inference APIs by default.

Highlights
- Pure Python backend (Flask) + vanilla JS frontend — all in this repo.
- Interpolated Kneser–Ney n-gram language model (train/save/load).
- Optional Transformer-backed inference (manually installed).
- BPE tokenizer with HF tokenizers fallback and pure-Python fallback.
- Dataset ingestion, file upload, zip extraction, deduplication, dataset viewer & export.
- Admin forced login (ENV VARS) + session revocation + CSRF protection.
- RAG (optional lazy load) with sentence-transformers + faiss-cpu if installed.
- Backups on every upload/train/save; optional S3 archival if env provided.
- Keepalive endpoint and docs for cron-job.org to avoid Render idling.
- Lightweight pytest tests for tokenizer and n-gram model.

## Quickstart (local)
```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
export ADMIN_USER=admin
export ADMIN_PASS=changeme
export ADMIN_SESSION_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
python app.py
# Open http://127.0.0.1:5000
```

## Deploy to Render (summary)
- Create a new Web Service, link to this repo, set Python build, set `ADMIN_USER`, `ADMIN_PASS`, `ADMIN_SESSION_SECRET` in Render dashboard, add `render.yaml` and `Procfile` included.
- See full deploy instructions below in this README for environment variables, optional features, and cron-job keepalive instructions.

## Files included (high level)
- `app.py` — Flask backend serving APIs and pages.
- `static/` — frontend JS/CSS.
- `data/uploads/` — sample dataset (JSONL ~3–5MB) and other uploads.
- `scripts/generate_groq.py` — example synthetic generator (not executed automatically).
- `models/` — model save/load area.
- `tests/` — pytest tests.
- `requirements.txt` and `requirements-optional.txt` (optional heavy deps).

-- README truncated here; full README included in repo.
# AI Assistant — Single-file deployable project
MIT License — see LICENSE.

This repository contains a fully functional AI assistant web application designed to run on Render's free tier without paid inference APIs by default.

Highlights
- Pure Python backend (Flask) + vanilla JS frontend — all in this repo.
- Interpolated Kneser–Ney n-gram language model (train/save/load).
- Optional Transformer-backed inference (manually installed).
- BPE tokenizer with HF tokenizers fallback and pure-Python fallback.
- Dataset ingestion, file upload, zip extraction, deduplication, dataset viewer & export.
- Admin forced login (ENV VARS) + session revocation + CSRF protection.
- RAG (optional lazy load) with sentence-transformers + faiss-cpu if installed.
- Backups on every upload/train/save; optional S3 archival if env provided.
- Keepalive endpoint and docs for cron-job.org to avoid Render idling.
- Lightweight pytest tests for tokenizer and n-gram model.

## Quickstart (local)
```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
export ADMIN_USER=admin
export ADMIN_PASS=changeme
export ADMIN_SESSION_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
python app.py
# Open http://127.0.0.1:5000
```

## Deploy to Render (summary)
- Create a new Web Service, link to this repo, set Python build, set `ADMIN_USER`, `ADMIN_PASS`, `ADMIN_SESSION_SECRET` in Render dashboard, add `render.yaml` and `Procfile` included.
- See full deploy instructions below in this README for environment variables, optional features, and cron-job keepalive instructions.

## Files included (high level)
- `app.py` — Flask backend serving APIs and pages.
- `static/` — frontend JS/CSS.
- `data/uploads/` — sample dataset (JSONL ~3–5MB) and other uploads.
- `scripts/generate_groq.py` — example synthetic generator (not executed automatically).
- `models/` — model save/load area.
- `tests/` — pytest tests.
- `requirements.txt` and `requirements-optional.txt` (optional heavy deps).

-- README truncated here; full README included in repo.
# AI Assistant — Single-file deployable project
MIT License — see LICENSE.

This repository contains a fully functional AI assistant web application designed to run on Render's free tier without paid inference APIs by default.

Highlights
- Pure Python backend (Flask) + vanilla JS frontend — all in this repo.
- Interpolated Kneser–Ney n-gram language model (train/save/load).
- Optional Transformer-backed inference (manually installed).
- BPE tokenizer with HF tokenizers fallback and pure-Python fallback.
- Dataset ingestion, file upload, zip extraction, deduplication, dataset viewer & export.
- Admin forced login (ENV VARS) + session revocation + CSRF protection.
- RAG (optional lazy load) with sentence-transformers + faiss-cpu if installed.
- Backups on every upload/train/save; optional S3 archival if env provided.
- Keepalive endpoint and docs for cron-job.org to avoid Render idling.
- Lightweight pytest tests for tokenizer and n-gram model.

## Quickstart (local)
```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
export ADMIN_USER=admin
export ADMIN_PASS=changeme
export ADMIN_SESSION_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
python app.py
# Open http://127.0.0.1:5000
```

## Deploy to Render (summary)
- Create a new Web Service, link to this repo, set Python build, set `ADMIN_USER`, `ADMIN_PASS`, `ADMIN_SESSION_SECRET` in Render dashboard, add `render.yaml` and `Procfile` included.
- See full deploy instructions below in this README for environment variables, optional features, and cron-job keepalive instructions.

## Files included (high level)
- `app.py` — Flask backend serving APIs and pages.
- `static/` — frontend JS/CSS.
- `data/uploads/` — sample dataset (JSONL ~3–5MB) and other uploads.
- `scripts/generate_groq.py` — example synthetic generator (not executed automatically).
- `models/` — model save/load area.
- `tests/` — pytest tests.
- `requirements.txt` and `requirements-optional.txt` (optional heavy deps).

-- README truncated here; full README included in repo.
