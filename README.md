# AI Assistant — Single-file deployable project

MIT License — see LICENSE.

This repository contains a fully functional AI assistant web application designed to run on **Render's free tier** without paid inference APIs by default.

### Highlights
- Pure **Python backend (Flask)** + **vanilla JS frontend**.
- Interpolated **Kneser–Ney n-gram language model** (train/save/load).
- Optional **Transformer-backed inference** (manual install).
- BPE tokenizer with HF tokenizers fallback and a pure-Python fallback.
- **Dataset ingestion** with file upload, zip extraction, deduplication, viewing, and export.
- Admin forced login with **environment variables**, session revocation, and CSRF protection.
- Optional **RAG (Retrieval-Augmented Generation)** with lazy loading.
- Backups on every upload/train/save, with optional S3 archival.
- Keepalive endpoint to prevent Render idling.
- Lightweight **pytest tests** for the tokenizer and n-gram model.

---

## Quickstart (local)

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
export ADMIN_USER=admin
export ADMIN_PASS=changeme
export ADMIN_SESSION_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
python app.py
# Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

