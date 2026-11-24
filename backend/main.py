"""ResolveAI — FastAPI application.
Single file: config from env, all routes, serves the built React frontend.
Run: uvicorn main:app --reload --port 8000
"""
import json, logging, os, secrets
from contextlib import asynccontextmanager
from pathlib import Path

# ---- tiny .env loader (no dotenv package) ----
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import nlp, db

log = logging.getLogger("resolveai")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---- constants ----
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
PORT = int(os.environ.get("PORT", 8000))

# ===================== STARTUP =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()                       # create the table on first start
    nlp.get_model()                    # load model into memory once
    nlp.get_ner()                      # initialise the NER pipeline
    nlp.analyze("Wi-Fi is not working in Cedar Hall.", with_reply=False, include_explanation=False)
    nlp.log_gemini_configuration()
    log.info("ResolveAI server ready — model loaded, DB initialised")
    yield

app = FastAPI(title="ResolveAI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["*"])

# ===================== ERROR HANDLERS =====================
@app.exception_handler(nlp.FileError)
async def file_error(_: Request, e: nlp.FileError):
    return JSONResponse({"detail": str(e)}, status_code=422)

@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, e: RequestValidationError):
    msg = "; ".join(str(x["msg"]).replace("Value error, ", "") for x in e.errors())
    return JSONResponse({"detail": msg}, status_code=422)

@app.exception_handler(Exception)
async def unexpected(_: Request, e: Exception):
    log.exception("Unhandled error")
    return JSONResponse({"detail": "Something went wrong on the server. Please try again."}, status_code=500)

# ===================== HEALTH =====================
def require_admin_diagnostic(request: Request):
    expected = os.environ.get("ADMIN_DIAGNOSTIC_TOKEN", "").strip()
    if not expected:
        raise HTTPException(503, "Admin Gemini diagnostics are disabled; configure ADMIN_DIAGNOSTIC_TOKEN.")
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "Administrator authentication required.", headers={"WWW-Authenticate": "Bearer"})


@app.post("/api/admin/gemini-test", dependencies=[Depends(require_admin_diagnostic)])
def admin_gemini_test():
    """Send an isolated authenticated test prompt directly to Gemini."""
    return nlp.run_gemini_diagnostic()


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_configured": nlp.gemini_available(), "gemini": nlp.gemini_diagnostics(),
            "ner": nlp.get_ner()[1]}

# ===================== SAMPLES =====================
@app.get("/api/samples")
def list_samples():
    kinds = {"txt": "Letter (TXT)", "pdf": "Letter (PDF)", "csv": "Batch (CSV)"}
    return [{"name": p.name, "label": f"{kinds[p.suffix[1:]]} · {p.name}"}
            for p in sorted(nlp.SAMPLES_DIR.iterdir()) if p.suffix[1:] in kinds]

@app.get("/api/samples/{name}")
def get_sample(name: str):
    safe = Path(name).name
    p = nlp.SAMPLES_DIR / safe
    if not p.is_file() or p.parent != nlp.SAMPLES_DIR:
        raise HTTPException(404, "Sample not found.")
    return nlp.extract_file(p.name, p.read_bytes())

# ===================== FILE EXTRACTION =====================
@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    raw = await file.read(nlp.MAX_FILE_BYTES + 1)
    return nlp.extract_file(file.filename or "", raw)

# ===================== ANALYSIS =====================
def _validate_text(text: str) -> str:
    text = " ".join(text.split())
    if len(text) < 5:
        raise HTTPException(422, "Please enter at least 5 characters.")
    if len(text) > nlp.MAX_TEXT_CHARS:
        raise HTTPException(422, f"Text is too long (max {nlp.MAX_TEXT_CHARS} characters).")
    return text

@app.post("/api/analyze")
def analyze(body: dict):
    text = _validate_text(body.get("text", ""))
    result = nlp.analyze(text, with_reply=True)
    result["status"] = "pending"
    result["resolved_at"] = None
    result["id"] = db.save(result)
    try:
        result["similar"] = nlp.find_similar(result["english_text"], db.get_recent_for_similarity(200), result["id"], max_seconds=1.5)
    except Exception:
        log.exception("Similar grievance lookup failed")
        result["similar"] = []
    return result

@app.post("/api/analyze/batch")
def analyze_batch(body: dict):
    texts = body.get("texts", [])
    texts = [" ".join(t.split())[:nlp.MAX_TEXT_CHARS] for t in texts if t and t.strip()]
    if not texts:
        raise HTTPException(422, "No grievances to analyse.")
    if len(texts) > nlp.MAX_BATCH_ROWS:
        raise HTTPException(422, f"Too many rows (max {nlp.MAX_BATCH_ROWS}).")
    out = []
    translated_count = 0
    for t in texts:
        lang = nlp.detect_language(t)
        if lang == "en":
            translation = (t, lang, None)
        elif translated_count < 10:
            translated_count += 1
            translation = nlp.to_english(t, allow_gemini=False, detected_language=lang)
        else:
            translation = (t, lang, "unavailable (original text used)")
        r = nlp.analyze(t, with_reply=False, include_explanation=False, translation=translation)
        r["status"] = "pending"
        r["resolved_at"] = None
        r["similar"] = []
        out.append(r)
    for r, gid in zip(out, db.save_many(out)):
        r["id"] = gid
    by = lambda k: {v: sum(1 for r in out if r[k] == v) for v in sorted({r[k] for r in out})}
    return {"count": len(out), "by_priority": by("priority"), "by_category": by("category"), "results": out}

# ===================== HISTORY =====================
@app.get("/api/history")
def history(limit: int = 100, status: str | None = None):
    if status is not None and status not in ("pending", "resolved"):
        raise HTTPException(422, "Status must be 'pending' or 'resolved'.")
    rows = db.get_history(limit)
    return [row for row in rows if status is None or row["status"] == status]

@app.patch("/api/history/{gid}")
def update_history_status(gid: int, body: dict):
    status = body.get("status")
    if status not in ("pending", "resolved"):
        raise HTTPException(422, "Status must be 'pending' or 'resolved'.")
    if not db.update_status(gid, status):
        raise HTTPException(404, "Record not found.")
    return {"id": gid, "status": status}

@app.patch("/api/history")
def update_history_status_many(body: dict):
    ids, status = body.get("ids"), body.get("status")
    if status not in ("pending", "resolved"):
        raise HTTPException(422, "Status must be 'pending' or 'resolved'.")
    if not isinstance(ids, list) or len(ids) > 200 or any(not isinstance(gid, int) or isinstance(gid, bool) for gid in ids):
        raise HTTPException(422, "Provide up to 200 grievance IDs.")
    return {"updated": db.update_status_many(ids, status), "status": status}

@app.delete("/api/history/{gid}")
def delete_one(gid: int):
    if not db.delete_one(gid):
        raise HTTPException(404, "Record not found.")
    return {"deleted": gid}

@app.delete("/api/history")
def clear_history():
    n = db.delete_all()
    return {"deleted": n}

# ===================== METRICS =====================
@app.get("/api/metrics")
def metrics():
    f = nlp.ARTIFACTS / "metrics.json"
    if not f.exists():
        raise HTTPException(404, "Metrics not available.")
    return json.loads(f.read_text())

@app.get("/api/similar")
def similar(id: int):
    row = db.get_by_id(id)
    if row is None:
        raise HTTPException(404, "Record not found.")
    return nlp.find_similar(row["english_text"], db.get_recent_for_similarity(500), id)

@app.get("/api/stats")
def stats():
    return db.get_stats()

# ===================== PIPELINE STEPS (Walkthrough) =====================
@app.get("/api/steps")
def steps(text: str = Query(..., min_length=5, max_length=2000)):
    """Return the intermediate output of every pipeline step for one text."""
    # 1. Language + translation
    english, lang, svc = nlp.to_english(text)
    # 2. Cleaning steps
    cs = nlp.clean_steps(english)
    # 3. TF-IDF top terms using PortableModel.vectorize (top 8 by weight, NumPy only)
    top_terms = nlp.get_top_tfidf_terms(cs["cleaned"], top_n=8)
    # 4. Classification with probabilities
    cls = nlp.classify(english)
    # 5. Sentiment
    sen, score = nlp.sentiment(english)
    # 6. Entities
    ents = nlp.extract_entities(english)
    # 7. Exact LLM prompt
    prompt = nlp.build_prompt(english, cls["category"], cls["priority"], sen, ents)

    return {
        "language": lang, "translated_by": svc, "english_text": english,
        "clean_steps": cs, "top_tfidf_terms": top_terms,
        "category": cls["category"], "priority": cls["priority"],
        "confidence": cls["confidence"],
        "category_probs": cls["category_probs"],
        "priority_probs": cls["priority_probs"],
        "sentiment": sen, "sentiment_score": score,
        "entities": ents, "prompt": prompt,
    }

# ===================== SOURCE CODE (Walkthrough) =====================
@app.get("/api/source/{name}")
def source(name: str):
    """Return the source code of a whitelisted function for the Walkthrough."""
    code = nlp.get_source(name)
    if code is None:
        raise HTTPException(404, f"Unknown function: {name}")
    return {"name": name, "source": code}

# ===================== SENTIMENT CHECK (Walkthrough) =====================
@app.get("/api/sentiment-check")
def sentiment_check(n: int = Query(20, ge=1, le=50)):
    """Return n random grievances from heldout.csv with their VADER label for the self-check exercise."""
    rows = nlp.get_held_out_rows(n)
    return [{"text": r["complaint_text"], "vader_label": nlp.sentiment(r["complaint_text"])[0], "vader_score": nlp.sentiment(r["complaint_text"])[1]} for r in rows]

# ===================== PROMPT COMPARISON (Walkthrough) =====================
@app.post("/api/prompt-compare")
def prompt_compare():
    """Run Prompt V1 and V2 on 3 random grievances from heldout.csv. Returns both outputs."""
    if not nlp.get_api_key():
        raise HTTPException(400, "Gemini API key is not configured. Set GEMINI_API_KEY to use this feature.")
    rows = nlp.get_held_out_rows(3)
    results = []
    for r in rows:
        t = r["complaint_text"]
        cls = nlp.classify(t)
        sen, _ = nlp.sentiment(t)
        ents = nlp.extract_entities(t)
        p1 = nlp.PROMPT_V1.format(g=t)
        p2 = nlp.build_prompt(t, cls["category"], cls["priority"], sen, ents)
        try:
            r1_text, _ = nlp.call_gemini(p1, timeout=25)
        except Exception as e:
            r1_text = f"Error: {e}"
        try:
            r2_text, _ = nlp.call_gemini(p2, as_json=True, timeout=25)
            try:
                json.loads(r2_text)
                v2_valid = True
            except Exception:
                v2_valid = False
        except Exception as e:
            r2_text = f"Error: {e}"
            v2_valid = False
        results.append({
            "text": t, "prompt_v1": p1, "prompt_v2": p2,
            "response_v1": r1_text, "response_v2": r2_text,
            "v2_valid_json": v2_valid,
        })
    return results

# ===================== SERVE FRONTEND =====================
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="spa")
