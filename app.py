
import os, io, json, zipfile, uuid, time, math, datetime, typing, httpx, re, pickle
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session as SASession
import numpy as np

# ------------------ DB Setup ------------------
APP_ENV = os.getenv("APP_ENV", "development")
DB_URL = os.getenv("DB_URL", "sqlite:///./data/app.db")
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/indices", exist_ok=True)
os.makedirs("data/tokenizer", exist_ok=True)
os.makedirs("data/models", exist_ok=True)

engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Chat(Base):
    __tablename__ = "chats"
    id = Column(String, primary_key=True)
    title = Column(String, default="Chat")
    rating = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(String, ForeignKey("chats.id"))
    role = Column(String)  # user|assistant|system
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")

class DatasetRow(Base):
    __tablename__ = "dataset_rows"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String)
    text = Column(Text)
    ext_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# -------------------- Admin auth, CSRF, and revocation --------------------
ADMIN_USER = os.getenv("ADMIN_USER","admin")
ADMIN_PASS = os.getenv("ADMIN_PASS","password")
ADMIN_SESSION_SECRET = os.getenv("ADMIN_SESSION_SECRET","please-change-this-secret")
_SIGNER = TimestampSigner(ADMIN_SESSION_SECRET)
# Simple revocation list persisted to file
REVOC_FILE = os.path.join("data","admin_revoked.txt")
os.makedirs("data", exist_ok=True)
_revoked = set()
if os.path.exists(REVOC_FILE):
    try:
        with open(REVOC_FILE,"r",encoding="utf-8") as f:
            for ln in f.read().splitlines():
                if ln.strip(): _revoked.add(ln.strip())
    except Exception:
        _revoked = set()

def persist_revocation(token_id: str):
    _revoked.add(token_id)
    try:
        with open(REVOC_FILE,"a",encoding="utf-8") as f:
            f.write(token_id + "\n")
    except Exception:
        pass

def make_admin_session():
    session_id = secrets.token_urlsafe(40)
    signed = _SIGNER.sign(session_id.encode()).decode()
    return signed, session_id

def verify_admin_session(signed_token: str, max_age: int = 60*60*24):
    try:
        unsigned = _SIGNER.unsign(signed_token.encode(), max_age=max_age).decode()
    except BadSignature:
        return False, None
    if unsigned in _revoked:
        return False, None
    return True, unsigned

# CSRF: issue a CSRF token stored in cookie (double-submit)
def make_csrf_token():
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")

def verify_csrf(cookie_token: str, header_token: str):
    if not cookie_token or not header_token: return False
    return hmac.compare_digest(cookie_token, header_token)

# -------------------- Backups (local + optional S3/GCS) --------------------
BACKUP_DIR = os.path.join("data","backups")
os.makedirs(BACKUP_DIR, exist_ok=True)
# optional S3 env
S3_BUCKET = os.getenv("S3_BUCKET","")
S3_KEY = os.getenv("S3_KEY","")
S3_SECRET = os.getenv("S3_SECRET","")
S3_REGION = os.getenv("S3_REGION","us-east-1")

def backup_to_local(db_session):
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = os.path.join(BACKUP_DIR, f"backup_{ts}.json")
    rows = db_session.query(DatasetRow).order_by(DatasetRow.created_at.asc()).all()
    chats = db_session.query(Chat).order_by(Chat.created_at.asc()).all()
    out = {"timestamp": ts, "rows": [], "chats": []}
    for r in rows:
        out["rows"].append({"id": r.id, "source": r.source, "text": r.text, "created_at": r.created_at.isoformat()})
    for c in chats:
        out["chats"].append({"id": c.id, "rating": c.rating, "created_at": c.created_at.isoformat(),
                             "messages": [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in c.messages]})
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # zip uploads directory
    upath = os.path.join("data","uploads")
    zipname = os.path.join(BACKUP_DIR, f"uploads_{ts}.zip")
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        if os.path.exists(upath):
            for root,_,files in os.walk(upath):
                for fn in files:
                    z.write(os.path.join(root,fn), os.path.relpath(os.path.join(root,fn), upath))
    return fname, zipname

def backup_maybe_s3(local_json, local_zip):
    # upload if S3 creds present and boto3 is available
    if not _HAS_BOTO3 or not S3_BUCKET or not S3_KEY or not S3_SECRET:
        return False
    try:
        sess = boto3.session.Session(aws_access_key_id=S3_KEY, aws_secret_access_key=S3_SECRET, region_name=S3_REGION)
        s3 = sess.client("s3")
        s3.upload_file(local_json, S3_BUCKET, os.path.basename(local_json))
        s3.upload_file(local_zip, S3_BUCKET, os.path.basename(local_zip))
        return True
    except Exception as e:
        print("S3 backup failed:", e)
        return False

# helper wrapper
def create_backup(db_session):
    try:
        local_json, local_zip = backup_to_local(db_session)
        backup_maybe_s3(local_json, local_zip)
    except Exception as e:
        print("backup error:", e)


# ------------------ Simple RAG (FAISS + ST) ------------------
def lazy_import_st():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer
def lazy_import_faiss():
    import faiss
    return faiss

class RAGIndex:
    def __init__(self):
        self.model_name = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = None
        self.index = None
        self.texts: List[str] = []
        self.ids: List[str] = []
        self.path_dir = "data/indices"
        self._load()

    def _load_embedder(self):
        if self.model is None:
            ST = lazy_import_st()
            self.model = ST(self.model_name)

    def _load(self):
        meta_path = os.path.join(self.path_dir, "meta.json")
        index_path = os.path.join(self.path_dir, "faiss.index")
        if os.path.exists(meta_path) and os.path.exists(index_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.texts = meta.get("texts", [])
            self.ids = meta.get("ids", [])
            faiss = lazy_import_faiss()
            self.index = faiss.read_index(index_path)

    def _save(self):
        with open(os.path.join(self.path_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "ids": self.ids}, f)
        if self.index is not None:
            faiss = lazy_import_faiss()
            faiss.write_index(self.index, os.path.join(self.path_dir, "faiss.index"))

    def add(self, text: str) -> str:
        self._load_embedder()
        emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        if self.index is None:
            faiss = lazy_import_faiss()
            self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        rid = str(len(self.texts))
        self.texts.append(text)
        self.ids.append(rid)
        self._save()
        return rid

    def rebuild(self):
        self._load_embedder()
        faiss = lazy_import_faiss()
        if not self.texts:
            self.index = None
            self._save()
            return 0
        emb = self.model.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self._save()
        return len(self.texts)

    def search(self, query: str, k: int = 8):
        if not self.texts:
            return []
        self._load_embedder()
        if self.index is None: self.rebuild()
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        D,I = self.index.search(q, min(k, len(self.texts)))
        out=[]
        for d,i in zip(D[0], I[0]):
            if i==-1: continue
            out.append({"id": self.ids[i], "text": self.texts[i], "score": float(d)})
        return out

rag = RAGIndex()

# ------------------ From-Scratch N-gram LM (Interpolated Kneser–Ney) ------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

def tokenize(text: str):
    return TOKEN_RE.findall(text)

class KneserNeyLM:
    """
    Interpolated Kneser–Ney word-level LM.
    Stores counts up to N=5 (configurable). Provides sampling.
    """
    def __init__(self, N: int = 5, d: float = 0.75):
        self.N = N
        self.d = d
        self.ngram_counts: List[Dict[tuple,int]] = [dict() for _ in range(N+1)]  # 0 unused
        self.context_counts: List[Dict[tuple,int]] = [dict() for _ in range(N+1)]
        self.continuation_counts: Dict[str,int] = {}  # unique contexts in which token appears
        self.vocab = set()
        self.trained = False

    def _add_count(self, ngram: tuple):
        n = len(ngram)
        self.ngram_counts[n][ngram] = self.ngram_counts[n].get(ngram,0)+1
        if n>1:
            ctx = ngram[:-1]
            self.context_counts[n][ctx] = self.context_counts[n].get(ctx,0)+1

    def fit(self, texts: List[str]):
        BOS = "<s>"; EOS = "</s>"
        seen_contexts_for_word = {}
        for text in texts:
            toks = [BOS]*(self.N-1) + tokenize(text) + [EOS]
            for t in toks:
                self.vocab.add(t)
            for i in range(self.N-1, len(toks)):
                for n in range(1, self.N+1):
                    if i-n+1 < 0: break
                    ngram = tuple(toks[i-n+1:i+1])
                    self._add_count(ngram)
                # track continuation counts
            for i in range(self.N-1, len(toks)):
                w = toks[i]
                ctx = tuple(toks[max(0,i-self.N+1):i])
                seen_contexts_for_word.setdefault(w, set()).add(ctx)
        for w, ctxs in seen_contexts_for_word.items():
            self.continuation_counts[w] = len(ctxs)
        self.total_tokens = sum(self.ngram_counts[1].values())
        self.trained = True

    def prob(self, ctx: tuple, w: str):
        # Modified KN recursion
        if not self.trained:
            return 1.0 / max(1, len(self.vocab))
        N = min(self.N-1, len(ctx))
        return self._p_rec(ctx[-N:], w, N)

    def _p_rec(self, ctx: tuple, w: str, n: int):
        # base: continuation prob
        if n == 0:
            Z = sum(self.continuation_counts.values())
            return (self.continuation_counts.get(w,0) or 1) / max(1, Z)
        counts_n = self.ngram_counts[n+1]
        ctx_counts = self.context_counts[n+1]
        c_ctx = ctx_counts.get(ctx, 0)
        c_ng = counts_n.get(ctx + (w,), 0)
        d = self.d
        # lambda backoff weight
        # number of unique continuations after ctx:
        T = len({ng[-1] for ng in counts_n.keys() if ng[:-1]==ctx})
        lambda_ctx = (d * T) / max(1, c_ctx) if c_ctx>0 else 1.0
        # max(.,0) discounted prob
        p_ml = max(c_ng - d, 0) / max(1, c_ctx) if c_ctx>0 else 0.0
        return p_ml + lambda_ctx * self._p_rec(ctx[1:] if len(ctx)>0 else tuple(), w, n-1)

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 1.0, top_k: int = 40):
        BOS = "<s>"; EOS="</s>"
        toks = [BOS]*(self.N-1) + tokenize(prompt)
        out = []
        for _ in range(max_new_tokens):
            ctx = tuple(toks[-(self.N-1):])
            # score all vocab (limit for speed)
            candidates = list(self.vocab)
            # compute probs
            probs = [self.prob(ctx, w) for w in candidates]
            # temperature + top-k
            if top_k and top_k < len(candidates):
                # pick top_k by prob
                pairs = sorted(zip(candidates, probs), key=lambda x:x[1], reverse=True)[:top_k]
                candidates, probs = zip(*pairs)
                candidates=list(candidates); probs=list(probs)
            # temperature softmax
            ps = np.array(probs, dtype=np.float64) + 1e-12
            ps = np.log(ps); ps = ps / max(1e-6, temperature); ps = np.exp(ps - np.max(ps))
            ps = ps / np.sum(ps)
            choice = np.random.choice(len(candidates), p=ps)
            w = candidates[choice]
            if w == EOS: break
            out.append(w); toks.append(w)
        # detokenize (simple spacing)
        text = ""
        for t in out:
            if re.match(r"^[A-Za-z0-9_]+$", t):
                text += (" " if text else "") + t
            else:
                text += t
        return text.strip()

# Global model
LM_PATH = "data/models/ngram_kn5.pkl"
lm = KneserNeyLM(N=5, d=0.75)

def load_model_if_exists():
    if os.path.exists(LM_PATH):
        with open(LM_PATH, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, KneserNeyLM)
        return obj
    return None

# ------------------ FastAPI ------------------

import hmac, hashlib, secrets, threading, time, base64, os
try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    _HAS_BOTO3 = False
from itsdangerous import TimestampSigner, BadSignature
app = FastAPI(title="From-Scratch AI (KN N-gram)", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# Schemas
class ChatRequest(BaseModel):
    chat_id: Optional[str] = None
    messages: List[Dict[str,str]]
    rating: Optional[int] = None
    max_new_tokens: Optional[int] = 128

class ChatResponse(BaseModel):
    chat_id: str
    reply: str
    used_rag_docs: List[Dict[str,Any]] = []

class SynthesizeRequest(BaseModel):
    prompt: str
    n_samples: int = Field(default=10, ge=1, le=500)

# Helpers
def extract_texts_from_jsonlike(obj):
    texts=[]
    if isinstance(obj, str):
        texts.append(obj)
    elif isinstance(obj, list):
        for x in obj: texts += extract_texts_from_jsonlike(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, str): texts.append(v)
    return texts

# Pages
@app.get("/", response_class=HTMLResponse, tags=["pages"])
def page_index():
    return HTMLResponse(open("static/index.html","r",encoding="utf-8").read())

@app.get("/admin", response_class=HTMLResponse, tags=["pages"])
def page_admin():
    return HTMLResponse(open("static/admin.html","r",encoding="utf-8").read())

@app.get("/data", response_class=HTMLResponse, tags=["pages"])
def page_data():
    return HTMLResponse(open("static/data.html","r",encoding="utf-8").read())

@app.get("/analytics", response_class=HTMLResponse, tags=["pages"])
def page_analytics():
    return HTMLResponse(open("static/analytics.html","r",encoding="utf-8").read())

@app.get("/keepalive", response_class=HTMLResponse, tags=["pages"])
def keepalive_page():
    return HTMLResponse(open("static/keepalive.html","r",encoding="utf-8").read())

@app.get("/api/keepalive")
def keepalive(): return {"ok": True, "ts": time.time()}

# Chat (uses RAG + local LM)


# -------------------- Admin endpoints --------------------
from fastapi import Cookie, Header

@app.post("/api/admin/login")
def admin_login(creds: dict):
    username = creds.get("user") or creds.get("username")
    password = creds.get("pass") or creds.get("password")
    if not username or not password:
        raise HTTPException(400, "missing credentials")
    if not hmac.compare_digest(username, ADMIN_USER) or not hmac.compare_digest(password, ADMIN_PASS):
        raise HTTPException(401, "invalid credentials")
    signed, sid = make_admin_session()
    resp = {"ok": True}
    response = Response(content=json.dumps(resp), media_type="application/json")
    # set session cookie and CSRF cookie
    csrf = make_csrf_token()
    response.set_cookie("admin_session", signed, httponly=True, samesite="Lax", max_age=60*60*24)
    response.set_cookie("csrf_token", csrf, httponly=False, samesite="Lax", max_age=60*60*24)
    return response

@app.post("/api/admin/logout")
def admin_logout(request: Request):
    # CSRF protection: require header 'x-csrf-token' match cookie 'csrf_token'
    header_csrf = None
    try:
        header_csrf = request.headers.get("x-csrf-token")
    except Exception:
        header_csrf = None
    cookie_csrf = request.cookies.get("csrf_token")
    if not verify_csrf(cookie_csrf, header_csrf):
        raise HTTPException(403, "CSRF check failed")
    token = request.cookies.get("admin_session")
    if token:
        ok, uid = verify_admin_session(token)
        if ok and uid:
            persist_revocation(uid)
    response = Response(content=json.dumps({"ok": True}), media_type="application/json")
    response.delete_cookie("admin_session"); response.delete_cookie("csrf_token")
    return response

@app.get("/api/admin/check")
def admin_check(request: Request):
    token = request.cookies.get("admin_session")
    ok, uid = verify_admin_session(token) if token else (False,None)
    if not ok:
        raise HTTPException(401, "unauthorized")
    return {"ok": True}
@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest):
    db: SASession = SessionLocal()
    chat_id = req.chat_id or str(uuid.uuid4())
    chat = db.query(Chat).filter(Chat.id==chat_id).first()
    if not chat:
        chat = Chat(id=chat_id); db.add(chat); db.commit()

@app.post("/api/admin/revoke")
def admin_revoke(request: Request):
    # CSRF protection: require header 'x-csrf-token' match cookie 'csrf_token'
    header_csrf = None
    try:
        header_csrf = request.headers.get("x-csrf-token")
    except Exception:
        header_csrf = None
    cookie_csrf = request.cookies.get("csrf_token")
    if not verify_csrf(cookie_csrf, header_csrf):
        raise HTTPException(403, "CSRF check failed")
    token = request.cookies.get("admin_session")
    ok, uid = verify_admin_session(token) if token else (False,None)
    if not ok or not uid:
        raise HTTPException(401, "unauthorized")
    persist_revocation(uid)
    return {"ok": True}


    for m in req.messages:
        if m.get("role") in ("user","system"):
            db.add(Message(chat_id=chat_id, role=m["role"], content=m.get("content","")))
    db.commit()

    last_user = next((m["content"] for m in reversed(req.messages) if m["role"]=="user"), "")
    # RAG
    used_docs=[]; ctx=""
    try:
        docs = rag.search(last_user, k=int(os.getenv("RAG_TOP_K","8")))
        used_docs = [{"id":d["id"], "score": d["score"], "text": d["text"][:600]} for d in docs]
        ctx = "\n".join([d["text"] for d in docs])[:int(os.getenv("MAX_CONTEXT_CHARS","16000"))]
    except Exception as e:
        used_docs=[{"error":"rag_failed","detail":str(e)}]

    prompt = (("Context:\n"+ctx+"\n\n") if ctx else "") + "User: " + last_user + "\nAssistant:"
    # Ensure model is trained
    global lm
    if not lm.trained:
        # Try load from disk
        inst = load_model_if_exists()
        if inst is not None:
            lm = inst
        else:
            # Minimal bootstrap on existing dataset rows as fallback
            texts=[r.text for r in SessionLocal().query(DatasetRow).limit(1000).all()]
            if not texts: texts=["Hello world.","How can I help you?"]
            lm.fit(texts)
    try:
        create_backup(db)
    except Exception:
        pass

    reply = lm.generate(prompt, max_new_tokens=int(req.max_new_tokens or 128), temperature=0.9, top_k=60)
    db.add(Message(chat_id=chat_id, role="assistant", content=reply))
    if req.rating: chat.rating = req.rating
    db.commit(); db.close()
    return ChatResponse(chat_id=chat_id, reply=reply, used_rag_docs=used_docs)

# Admin
@app.get("/api/chats", tags=["admin"])
def list_chats(request: Request, page: int=1, page_size: int=20):
    
    # --- admin auth check ---
    token = request.cookies.get("admin_session") if 'request' in locals() else None
    ok = False
    uid = None
    if token:
        ok, uid = verify_admin_session(token)
    if not ok:
        raise HTTPException(401, "unauthorized - admin required")
    db: SASession = SessionLocal()
    total = db.query(Chat).count()
    pages = max(1, math.ceil(total/max(1,page_size)))
    items = db.query(Chat).order_by(Chat.created_at.desc()).offset((page-1)*page_size).limit(page_size).all()
    out=[{"id":c.id,"title":c.title,"rating":c.rating,"created_at":c.created_at.isoformat()} for c in items]
    db.close()
    return {"page":page,"pages":pages,"total":total,"items":out}

@app.get("/api/messages", tags=["admin"])
def list_messages(request: Request, chat_id: str):
    
    # --- admin auth check ---
    token = request.cookies.get("admin_session") if 'request' in locals() else None
    ok = False
    uid = None
    if token:
        ok, uid = verify_admin_session(token)
    if not ok:
        raise HTTPException(401, "unauthorized - admin required")
    db: SASession = SessionLocal()
    msgs = db.query(Message).filter(Message.chat_id==chat_id).order_by(Message.created_at.asc()).all()
    out=[{"id":m.id,"role":m.role,"content":m.content,"created_at":m.created_at.isoformat()} for m in msgs]
    db.close()
    return out

# Data ingest
from fastapi import UploadFile, File, Form
@app.post("/api/data/upload", tags=["data"])
async def upload_data(mode: str = Form("auto"), file: UploadFile = File(...)):
    db: SASession = SessionLocal()
    raw_dir = "data/uploads"; os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, file.filename)
    with open(path, "wb") as f: f.write(await file.read())

    added=0
    def add_text(src, text):
        nonlocal added
        rid = rag.add(text)
        db.add(DatasetRow(source=src, text=text, ext_id=rid)); 

    def handle_jsonl_bytes(b: bytes, source: str):
        for line in b.decode("utf-8", errors="ignore").splitlines():
            s=line.strip()
            if not s: continue
            try: obj=json.loads(s)
            except Exception: continue
            if isinstance(obj, dict) and "input" in obj and "output" in obj:
                add_text(source, f"User: {obj['input']}\nAssistant: {obj['output']}")
            else:
                for t in extract_texts_from_jsonlike(obj): add_text(source, t)

    if file.filename.endswith(".zip"):
        with zipfile.ZipFile(path,"r") as z:
            for name in z.namelist():
                if not (name.endswith(".json") or name.endswith(".jsonl")): continue
                data=z.read(name)
                if name.endswith(".jsonl"): handle_jsonl_bytes(data, f"{file.filename}:{name}")
                else:
                    try: obj=json.loads(data.decode("utf-8",errors="ignore"))
                    except Exception: continue
                    items=obj if isinstance(obj,list) else [obj]
                    for it in items:
                        for t in extract_texts_from_jsonlike(it): add_text(f"{file.filename}:{name}", t); added+=1
    elif file.filename.endswith(".jsonl") or mode=="jsonl":
        with open(path,"rb") as f: handle_jsonl_bytes(f.read(), file.filename)
    else:
        try: obj=json.load(open(path,"r",encoding="utf-8"))
        except Exception as e:
            db.close(); raise HTTPException(400, f"invalid json: {e}")
        items=obj if isinstance(obj,list) else [obj]
        for it in items:
            for t in extract_texts_from_jsonlike(it): add_text(file.filename, t); added+=1

    db.commit(); rag.rebuild(); db.close()
    return {"ok": True}

@app.get("/api/data/list", tags=["data"])
def data_list(request: Request, ):
    
    # --- admin auth check ---
    token = request.cookies.get("admin_session") if 'request' in locals() else None
    ok = False
    uid = None
    if token:
        ok, uid = verify_admin_session(token)
    if not ok:
        raise HTTPException(401, "unauthorized - admin required")
    db: SASession = SessionLocal()
    rows = db.query(DatasetRow).order_by(DatasetRow.created_at.desc()).limit(500).all()
    out=[{"id":r.id,"source":r.source,"text":r.text[:800],"created_at":r.created_at.isoformat()} for r in rows]
    db.close(); return out

@app.post("/api/rag/reindex", tags=["rag"])
def rag_reindex(request: Request, ):
    
    # --- admin auth check ---
    token = request.cookies.get("admin_session") if 'request' in locals() else None
    ok = False
    uid = None
    if token:
        ok, uid = verify_admin_session(token)
    if not ok:
        raise HTTPException(401, "unauthorized - admin required")
    n=rag.rebuild(); return {"ok":True,"count":n}

# Groq synthetic data (only for generation of dataset)
class SynthesizeRequest(BaseModel):
    prompt: str
    n_samples: int = Field(default=10, ge=1, le=500)

@app.post("/api/data/groq_synthesize", tags=["data"])
def groq_synthesize(req: SynthesizeRequest):
    api_key=os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(400, "GROQ_API_KEY not set")
    model_name=os.getenv("GROQ_MODEL_FOR_SYNTH","llama-3.1-405b")
    base=os.getenv("GROQ_BASE_URL","https://api.groq.com")
    url=f"{base}/openai/v1/chat/completions"
    template=f"""Generate JSONL training pairs. Each line is a JSON object:
- "input": user message
- "output": assistant reply
- "metadata": {{"topic":"...","difficulty":"...","style":"..."}}
Prompt seed:
{req.prompt}
Each line must be a single JSON object.
"""
    out_lines=[]
    with httpx.Client(timeout=120) as client:
        for _ in range(req.n_samples):
            r = client.post(url, headers={"Authorization": f"Bearer {api_key}"}, json={
                "model": model_name, "messages":[{"role":"user","content":template}], "temperature": 0.9
            }); r.raise_for_status()
            content=r.json()["choices"][0]["message"]["content"]
            out_lines += [ln for ln in content.splitlines() if ln.strip().startswith("{")]
    # Store into dataset + RAG
    db: SASession = SessionLocal(); added=0
    for ln in out_lines:
        try:
            obj=json.loads(ln.strip())
            txt=f"User: {obj['input']}\nAssistant: {obj['output']}"
            rid=rag.add(txt)
            db.add(DatasetRow(source="groq_synth", text=txt, ext_id=rid)); added+=1
        except Exception:
            continue
    db.commit(); db.close()
    return {"ok": True, "added": added}

# Training API (builds/loads/saves KN model from DatasetRows)
@app.post("/api/train/start", tags=["train"])
def train_start(request: Request, ):
    
    # --- admin auth check ---
    token = request.cookies.get("admin_session") if 'request' in locals() else None
    ok = False
    uid = None
    if token:
        ok, uid = verify_admin_session(token)
    if not ok:
        raise HTTPException(401, "unauthorized - admin required")
    db: SASession = SessionLocal()
    rows = db.query(DatasetRow).order_by(DatasetRow.created_at.asc()).all()
    texts=[r.text for r in rows] if rows else []
    if not texts:
        return {"ok": False, "num_sentences": 0, "note": "No dataset rows to train on."}
    global lm
    lm = KneserNeyLM(N=5, d=0.75)
    lm.fit(texts)
    try:
        create_backup(db)
    except Exception:
        pass
    return {"ok": True, "num_sentences": len(texts), "vocab": len(lm.vocab)}

@app.post("/api/train/save", tags=["train"])
def train_save(request: Request, ):
    
    # --- admin auth check ---
    token = request.cookies.get("admin_session") if 'request' in locals() else None
    ok = False
    uid = None
    if token:
        ok, uid = verify_admin_session(token)
    if not ok:
        raise HTTPException(401, "unauthorized - admin required")
    global lm
    if not lm.trained: raise HTTPException(400, "Model not trained yet")
    with open(LM_PATH, "wb") as f:
        pickle.dump(lm, f)
    return {"ok": True, "path": LM_PATH}

@app.get("/api/train/load", tags=["train"])
def train_load(request: Request, ):
    
    # --- admin auth check ---
    token = request.cookies.get("admin_session") if 'request' in locals() else None
    ok = False
    uid = None
    if token:
        ok, uid = verify_admin_session(token)
    if not ok:
        raise HTTPException(401, "unauthorized - admin required")
    global lm
    inst = load_model_if_exists()
    if inst is None: return {"ok": False}
    lm = inst; return {"ok": True}

# Analytics & export
@app.get("/api/analytics/summary", tags=["analytics"])
def analytics_summary():
    db: SASession = SessionLocal()
    total_chats = db.query(func.count(Chat.id)).scalar() or 0
    total_messages = db.query(func.count(Message.id)).scalar() or 0
    ratings = [r[0] for r in db.query(Chat.rating).filter(Chat.rating.isnot(None)).all()]
    avg_rating = (sum(ratings)/len(ratings)) if ratings else None
    db.close(); return {"total_chats":total_chats,"total_messages":total_messages,"avg_rating":avg_rating}

@app.get("/api/analytics/ratings_hist", tags=["analytics"])
def ratings_hist():
    db: SASession = SessionLocal()
    counts = {i:0 for i in range(1,6)}
    for (r,) in db.query(Chat.rating).filter(Chat.rating.isnot(None)).all():
        if r in counts: counts[r]+=1
    db.close()
    labels=[str(i) for i in range(1,6)]; values=[counts[i] for i in range(1,6)]
    return {"labels":labels,"values":values}

@app.get("/api/analytics/top_queries", tags=["analytics"])
def top_queries(limit: int = 10):
    db: SASession = SessionLocal()
    msgs = db.query(Message).filter(Message.role=="user").order_by(Message.created_at.desc()).limit(1000).all()
    freq={}
    for m in msgs:
        key=(m.content or "").strip().split("\n")[0][:40]
        if not key: continue
        freq[key]=freq.get(key,0)+1
    items=sorted(freq.items(), key=lambda x:x[1], reverse=True)[:limit]
    labels=[k for k,_ in items]; values=[v for _,v in items]
    db.close(); return {"labels":labels,"values":values}

@app.get("/api/export/chats", tags=["export"])
def export_chats(request: Request, ):
    
    # --- admin auth check ---
    token = request.cookies.get("admin_session") if 'request' in locals() else None
    ok = False
    uid = None
    if token:
        ok, uid = verify_admin_session(token)
    if not ok:
        raise HTTPException(401, "unauthorized - admin required")
    db: SASession = SessionLocal()
    out=[]
    chats=db.query(Chat).order_by(Chat.created_at.asc()).all()
    for c in chats:
        for m in c.messages:
            out.append({"chat_id":c.id,"role":m.role,"content":m.content,"created_at":m.created_at.isoformat(),"rating":c.rating})
    db.close(); return out

@app.get("/api/export/dataset", tags=["export"])
def export_dataset(request: Request, ):
    
    # --- admin auth check ---
    token = request.cookies.get("admin_session") if 'request' in locals() else None
    ok = False
    uid = None
    if token:
        ok, uid = verify_admin_session(token)
    if not ok:
        raise HTTPException(401, "unauthorized - admin required")
    db: SASession = SessionLocal()
    rows=db.query(DatasetRow).order_by(DatasetRow.created_at.asc()).all()
    out=[{"id":r.id,"source":r.source,"text":r.text,"created_at":r.created_at.isoformat()} for r in rows]
    db.close(); return out


# -------------------- Full RAG (FAISS + SentenceTransformers) --------------------
def _lazy_load_st():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def _lazy_load_faiss():
    import faiss
    return faiss

class FullRAG:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None
        self._index = None
        self.texts = []
        self.ids = []
        self.meta_path = os.path.join("data","indices","meta.json")
        self.index_path = os.path.join("data","indices","faiss.index")
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        self._load()

    def _load_embed(self):
        if self._model is None:
            ST = _lazy_load_st()
            self._model = ST(self.model_name)

    def _load(self):
        if os.path.exists(self.meta_path) and os.path.exists(self.index_path):
            try:
                with open(self.meta_path,"r",encoding="utf-8") as f:
                    m = json.load(f)
                self.texts = m.get("texts", [])
                self.ids = m.get("ids", [])
                faiss = _lazy_load_faiss()
                self._index = faiss.read_index(self.index_path)
            except Exception as e:
                print("Failed to load RAG index:", e)
                self.texts = []; self.ids = []; self._index = None

    def save(self):
        try:
            if self._index is not None:
                faiss = _lazy_load_faiss()
                faiss.write_index(self._index, self.index_path)
            with open(self.meta_path,"w",encoding="utf-8") as f:
                json.dump({"texts": self.texts, "ids": self.ids}, f, ensure_ascii=False)
        except Exception as e:
            print("Failed to save RAG:", e)

    def add(self, text: str):
        self._load_embed()
        emb = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        if self._index is None:
            faiss = _lazy_load_faiss()
            d = emb.shape[1]
            self._index = faiss.IndexFlatIP(d)
        self._index.add(emb)
        rid = str(len(self.texts))
        self.texts.append(text); self.ids.append(rid)
        self.save()
        return rid

    def rebuild(self):
        self._load_embed()
        if not self.texts:
            self._index = None; self.save(); return 0
        emb = self._model.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        faiss = _lazy_load_faiss()
        d = emb.shape[1]
        self._index = faiss.IndexFlatIP(d)
        self._index.add(emb)
        self.save()
        return len(self.texts)

    def search(self, query: str, k: int = 8):
        if not self.texts:
            return []
        self._load_embed()
        if self._index is None:
            self.rebuild()
        q = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = self._index.search(q, min(k, len(self.texts)))
        out=[]
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            out.append({"id": self.ids[idx], "text": self.texts[idx], "score": float(score)})
        return out

# instantiate
try:
    RAG = FullRAG()
except Exception as e:
    print("RAG init failed:", e)
    class FallbackRAG:
        def add(self, text): return rag_add(text)
        def rebuild(self): return 0
        def search(self, q, k=8): return []
    RAG = FallbackRAG()

def rag_add(text): return RAG.add(text)
def rag_rebuild(): return RAG.rebuild()
def rag_search(q,k=8): return RAG.search(q,k)
