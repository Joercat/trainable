# From-Scratch AI — Render Free Tier (No External Inference)

**Updated:** 2025-08-13T23:58:56.620092 UTC

This build removes all hosted AI providers. The model that answers **is built from scratch inside this app**:
a CPU-friendly **Interpolated Kneser–Ney N-gram language model** (word-level), trained on your uploaded data or on synthetic data you generate using **Groq (generation only, never inference)**.

### Highlights
- **No OpenAI/Together/Groq inference**. The chat uses only the local N-gram LM.
- **Training**: upload datasets (JSON/JSONL/ZIP) or synthesize JSONL pairs with Groq; then train the N-gram LM via `/api/train/*` endpoints or the **Data** page.
- **RAG**: FAISS + sentence-transformers for retrieval; retrieved context is prepended to prompts before LM generation.
- **Tokenizer**: simple regex word+punct tokenizer, plus optional BPE training/export (tokenizers lib) for inspection.
- **Admin/Analytics**: ratings, exports, histograms, top queries.
- **Keepalive**: `/keepalive` page and `/api/keepalive` endpoint for cron.

### Deploy (Render)
1. Push to GitHub.
2. Render Web Service (free).
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn -k uvicorn.workers.UvicornWorker app:app --timeout 180 --workers 1 --threads 8 --bind 0.0.0.0:$PORT`
3. Env vars:
   - `APP_ENV=production`
   - `DB_URL=sqlite:///./data/app.db`
   - `EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`
   - `RAG_TOP_K=8`
   - `MAX_CONTEXT_CHARS=16000`
   - Optional: `GROQ_API_KEY` (for synthetic data only)

### Train & Use
- Upload data at **/data** (JSON/JSONL or ZIP of those).
- (Optional) Generate synthetic JSONL via Groq (still no Groq inference).
- Click **Train N-gram** on the Data page or call `POST /api/train/start`.
- The chat page (`/`) uses the trained model to respond. Sampling options at `/api/model/generate`.

### Files
- `app.py` — single-file FastAPI app with model, training, RAG, UI, admin.
- `static/` — chat/admin/data/analytics pages & JS.
- `data/` — uploads, tokenizer, indexes, model weights (.pkl).

MIT License.


## Backups
This project stores local backups under `data/backups/` by default. You can enable S3 backups by setting S3_BUCKET, S3_KEY, S3_SECRET, and S3_REGION environment variables. boto3 must be available for S3 uploads.
