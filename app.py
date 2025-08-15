
import os, io, zipfile, json, sqlite3, time, hashlib, secrets
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for, abort, g
from werkzeug.utils import secure_filename
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "app.db"

for d in (DATA_DIR, UPLOAD_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def get_db():
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db

# Initialize DB if missing
if not DB_PATH.exists():
    db = get_db()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS admin_sessions (session_id TEXT PRIMARY KEY, created_at REAL);
    CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL);
    CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, role TEXT, text TEXT, created_at REAL);
    CREATE TABLE IF NOT EXISTS dataset_rows (id INTEGER PRIMARY KEY AUTOINCREMENT, source_file TEXT, data TEXT, hash TEXT, created_at REAL);
    CREATE TABLE IF NOT EXISTS backups (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, created_at REAL);
    CREATE TABLE IF NOT EXISTS model_meta (name TEXT PRIMARY KEY, meta TEXT);
    CREATE TABLE IF NOT EXISTS ratings (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, message_id INTEGER, rating INTEGER, created_at REAL);
    """)
    db.commit()
    db.close()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB uploads safe-guard

# Simple CSRF: server stores token in session, clients must provide X-CSRF-Token on POST
def require_csrf():
    token = session.get("_csrf_token")
    header = request.headers.get("X-CSRF-Token")
    if not token or header != token:
        abort(403, "Invalid CSRF token")

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*a, **k):
        if not session.get("admin"):
            return jsonify({"error":"login required"}), 401
        return fn(*a, **k)
    return wrapper

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/admin")
def admin_page():
    return send_from_directory("templates", "admin.html")

@app.route("/data.html")
def data_page():
    return send_from_directory("templates", "data.html")

@app.route("/dataset.html")
def dataset_page():
    return send_from_directory("templates", "dataset.html")

@app.route("/keepalive.html")
def keepalive_page():
    return send_from_directory("templates", "keepalive.html")

# Static
@app.route("/static/<path:p>")
def static_files(p):
    return send_from_directory("static", p)

# Admin login/logout/check
@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    payload = request.json or {}
    user = payload.get("user")
    pwd = payload.get("pass")
    if not user or not pwd:
        return jsonify({"error":"missing"}), 400
    ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
    ADMIN_PASS = os.environ.get("ADMIN_PASS", "changeme")
    if user == ADMIN_USER and pwd == ADMIN_PASS:
        session["admin"] = True
        session["_csrf_token"] = secrets.token_hex(16)
        sid = secrets.token_hex(24)
        db = get_db()
        db.execute("INSERT INTO admin_sessions(session_id, created_at) VALUES (?, ?)", (sid, time.time()))
        db.commit()
        session["sid"] = sid
        return jsonify({"ok":True, "csrf": session["_csrf_token"]})
    return jsonify({"error":"invalid"}), 401

@app.route("/api/admin/logout", methods=["POST"])
def admin_logout():
    session_keys = list(session.keys())
    for k in session_keys:
        session.pop(k, None)
    return jsonify({"ok":True})

@app.route("/api/admin/check")
def admin_check():
    return jsonify({"admin": bool(session.get("admin"))})

# Keepalive endpoint
@app.route("/api/keepalive")
def api_keepalive():
    return jsonify({"ok":True, "ts": time.time()})

# Data discovery: list files in data/uploads
@app.route("/api/data/discover")
def data_discover():
    files = []
    for p in UPLOAD_DIR.iterdir():
        if p.is_file():
            files.append({"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime})
    return jsonify({"files": files})

# Data list rows
@app.route("/api/data/list")
def data_list():
    db = get_db()
    cur = db.execute("SELECT id, source_file, created_at FROM dataset_rows ORDER BY id DESC LIMIT 200")
    rows = [dict(r) for r in cur.fetchall()]
    return jsonify({"rows": rows})

# Upload handling
@app.route("/api/data/upload", methods=["POST"])
def data_upload():
    # require_csrf()  # optional: require CSRF token from admin UI; kept simple for demo
    f = request.files.get("file")
    if not f:
        return jsonify({"error":"no file"}), 400
    filename = secure_filename(f.filename)
    dest = UPLOAD_DIR / filename
    f.save(str(dest))
    # If zip, extract any .json/.jsonl into uploads
    if zipfile.is_zipfile(str(dest)):
        with zipfile.ZipFile(str(dest)) as z:
            for nm in z.namelist():
                if nm.lower().endswith((".json", ".jsonl")):
                    z.extract(nm, path=str(UPLOAD_DIR))
    # Auto-ingest discovered files into dataset_rows (dedupe by hash)
    ingested = 0
    db = get_db()
    for p in UPLOAD_DIR.iterdir():
        if p.suffix.lower() not in [".json", ".jsonl"]:
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                h = hashlib.sha256(ln.encode("utf-8")).hexdigest()
                cur = db.execute("SELECT 1 FROM dataset_rows WHERE hash=?", (h,)).fetchone()
                if cur:
                    continue
                db.execute("INSERT INTO dataset_rows(source_file, data, hash, created_at) VALUES (?, ?, ?, ?)",
                           (p.name, ln, h, time.time()))
                ingested += 1
    db.commit()
    return jsonify({"ok":True, "uploaded": filename, "ingested": ingested})

# Chat endpoint: stores conversation, runs the N-gram LM (or optional transformer if available)
@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.json or {}
    user_text = payload.get("message","")
    chat_id = None
    db = get_db()
    cur = db.execute("INSERT INTO chats(created_at) VALUES (?)", (time.time(),))
    chat_id = cur.lastrowid
    db.execute("INSERT INTO messages(chat_id, role, text, created_at) VALUES (?, ?, ?, ?)", (chat_id, "user", user_text, time.time()))
    db.commit()

    # Try to use trained InterpolatedKneserNey model if present, otherwise fallback to dataset lookup
    reply = ""
    model_path = MODELS_DIR / "ngram_model.json"
    meta_path = MODELS_DIR / "ngram.json"
    if model_path.exists():
        try:
            from app_utils import InterpolatedKneserNey
            m = InterpolatedKneserNey(n=3)
            m.load(str(model_path))
            # Use the input text as a prompt; generate reply using predict()
            r = m.predict(user_text, max_len=60)
            if r and len(r.strip())>0:
                reply = r
            else:
                # fallback to meta default reply
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    reply = meta.get("default_reply","I'm learning; please try rephrasing.")
                else:
                    reply = "I'm still learning. Could you rephrase?"
        except Exception as e:
            reply = "Model error: " + str(e)
    else:
        # fallback dataset substring match
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {"default_reply":"I'm still learning. Could you rephrase?"}
        best = None; best_score = 0
        db = get_db()
        cur = db.execute("SELECT id, data FROM dataset_rows ORDER BY id DESC LIMIT 8000")
        for r in cur.fetchall():
            try:
                d = json.loads(r["data"])
                q = d.get("input","") or d.get("question","") or ""
                a = d.get("response","") or d.get("answer","") or d.get("output","") or ""
                if not q or not a: continue
                # compute a simple word overlap score
                score = sum(1 for t in q.lower().split() if t in user_text.lower().split())
                if score>best_score:
                    best_score = score; best = a
            except Exception:
                continue
        if best and best_score>0:
            reply = best
        else:
            reply = meta.get("default_reply","I don't have a good answer yet. Try a different phrasing.")

    # store assistant message
    db.execute("INSERT INTO messages(chat_id, role, text, created_at) VALUES (?, ?, ?, ?)", (chat_id, "assistant", reply, time.time()))
    db.commit()

    return jsonify({"reply": reply, "chat_id": chat_id})
# store assistant message
    db.execute("INSERT INTO messages(chat_id, role, text, created_at) VALUES (?, ?, ?, ?)", (chat_id, "assistant", reply, time.time()))
    db.commit()

    return jsonify({"reply": reply, "chat_id": chat_id})

# Ratings endpoint (stored as part of chat flow)
@app.route("/api/rate", methods=["POST"])
def api_rate():
    payload = request.json or {}
    chat_id = payload.get("chat_id")
    message_id = payload.get("message_id")
    rating = int(payload.get("rating",0))
    db = get_db()
    db.execute("INSERT INTO ratings(chat_id, message_id, rating, created_at) VALUES (?, ?, ?, ?)", (chat_id, message_id, rating, time.time()))
    db.commit()
    return jsonify({"ok":True})

# Training endpoints: start/save/load
@app.route("/api/train/start", methods=["POST"])
@login_required
def train_start():
    require_csrf()
    payload = request.json or {}
    use_ratings = bool(payload.get("use_ratings"))
    model_type = payload.get("model","ngram")
    db = get_db()
    # gather pairs
    cur = db.execute("SELECT id, data FROM dataset_rows")
    pairs = []
    for r in cur.fetchall():
        try:
            d = json.loads(r["data"])
            inp = d.get("input") or d.get("question") or d.get("q") or ""
            out = d.get("response") or d.get("answer") or d.get("a") or ""
            if inp and out:
                pairs.append((inp.strip(), out.strip()))
        except Exception:
            continue
    # compute average rating to weight dataset (if ratings exist)
    avg_rating = 0.0
    rcur = db.execute("SELECT AVG(rating) as avg_rating FROM ratings").fetchone()
    if rcur and rcur["avg_rating"]:
        try:
            avg_rating = float(rcur["avg_rating"])
        except Exception:
            avg_rating = 0.0
    # rating multiplier: shifts dataset repetition based on positive feedback (range ~1.0 to ~2.0)
    rating_multiplier = 1.0 + max(0.0, (avg_rating - 3.0)) / 2.0
    # prepare training lines: for ngram we will combine pairs into "input ||| output" style lines
    train_lines = []
    for inp, out in pairs:
        # repeat each pair proportional to rating_multiplier to bias towards higher-rated historical dataset
        repeats = max(1, int(round(rating_multiplier)))
        for _ in range(repeats):
            train_lines.append((inp + " ||| " + out))
    # Deduplicate and sanitize
    from app_utils import dedupe_lines, InterpolatedKneserNey
    train_lines = dedupe_lines(train_lines)
    # create model meta and train ngram-like model
    model_meta = { "type": "ngram", "trained_at": time.time(), "default_reply": "Hello — I am your professional AI assistant. I provide concise, accurate, and courteous responses. How may I help you today?", "persona": "professional" }
    if model_type == "ngram":
        # Train an interpolated Kneser-Ney-like ngram model on the 'output' side to generate replies
        replies = [ln.split(' ||| ',1)[1] if ' ||| ' in ln else ln for ln in train_lines]
        m = InterpolatedKneserNey(n=3)
        m.train_lines(replies)
        # save model to models/ngram.json (use m.save to write counts)
        model_path = MODELS_DIR / "ngram_model.json"
        m.save(str(model_path))
        model_meta["model_file"] = str(model_path.name)
        model_meta["pairs_used"] = len(replies)
    # save metadata
    mp = MODELS_DIR / "ngram.json"
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(model_meta, fh)
    return jsonify({"ok":True, "pairs_used": model_meta.get("pairs_used",0), "rating_multiplier": rating_multiplier})
@, methods=["POST"]
@login_required
def train_save():
    require_csrf()
    payload = request.json or {}
    model_name = payload.get("name","ngram")
    meta_path = MODELS_DIR / f"{model_name}.json"
    # just mark meta stored; already saved during train_start
    return jsonify({"ok":True, "saved": str(meta_path.name)})

@app.route("/api/train/load", methods=["GET"])
@login_required
def train_load():
    name = request.args.get("model","ngram")
    mp = MODELS_DIR / f"{name}.json"
    if not mp.exists():
        return jsonify({"error":"missing"}), 404
    with open(mp, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return jsonify({"ok":True, "meta": meta})

# Backups
def create_backup():
    ts = int(time.time())
    fname = f"backup_{ts}.zip"
    p = DATA_DIR / fname
    with zipfile.ZipFile(str(p), "w", zipfile.ZIP_DEFLATED) as z:
        # dataset rows export
        db = get_db()
        cur = db.execute("SELECT data FROM dataset_rows")
        rows = [r["data"] for r in cur.fetchall()]
        z.writestr("dataset_snapshot.jsonl", "\n".join(rows))
        # uploads zip
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                z.write(str(f), arcname="uploads/"+f.name)
    db = get_db()
    db.execute("INSERT INTO backups(filename, created_at) VALUES (?, ?)", (fname, time.time()))
    db.commit()
    return str(fname)

@app.route("/api/backups/list")
@login_required
def backups_list():
    db = get_db()
    cur = db.execute("SELECT id, filename, created_at FROM backups ORDER BY id DESC LIMIT 50")
    return jsonify({"backups":[dict(r) for r in cur.fetchall()]})

@app.route("/api/backups/restore", methods=["POST"])
@login_required
def backups_restore():
    require_csrf()
    payload = request.json or {}
    fname = payload.get("filename")
    if not fname:
        return jsonify({"error":"missing"}), 400
    p = DATA_DIR / fname
    if not p.exists():
        return jsonify({"error":"not found"}), 404
    with zipfile.ZipFile(str(p)) as z:
        if "dataset_snapshot.jsonl" in z.namelist():
            data = z.read("dataset_snapshot.jsonl").decode("utf-8").splitlines()
            db = get_db()
            for ln in data:
                h = hashlib.sha256(ln.encode("utf-8")).hexdigest()
                db.execute("INSERT OR IGNORE INTO dataset_rows(source_file, data, hash, created_at) VALUES (?, ?, ?, ?)",
                           ("backup_restore", ln, h, time.time()))
            db.commit()
    return jsonify({"ok":True})

# Analytics
@app.route("/api/analytics/summary")
@login_required
def analytics_summary():
    db = get_db()
    cur = db.execute("SELECT COUNT(*) as cnt FROM chats").fetchone()
    chat_count = cur["cnt"]
    r = db.execute("SELECT AVG(rating) as avg_rating FROM ratings").fetchone()
    avg_rating = r["avg_rating"] or 0
    return jsonify({"chats": chat_count, "avg_rating": avg_rating})

@app.route("/api/analytics/ratings")
@login_required
def analytics_ratings():
    db = get_db()
    cur = db.execute("SELECT rating, COUNT(*) as cnt FROM ratings GROUP BY rating").fetchall()
    return jsonify({"ratings":[dict(r) for r in cur]})

if __name__ == "__main__":
    # Use secret from ENV for sessions
    s = os.environ.get("ADMIN_SESSION_SECRET") or "dev_secret_change"
    app.secret_key = s
    # create an initial backup for included dataset
    try:
        create_backup()
    except Exception:
        pass
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
