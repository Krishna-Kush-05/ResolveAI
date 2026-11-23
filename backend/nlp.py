"""NLP core for ResolveAI.
Sections: cleaning, classification (PortableModel with NumPy), sentiment,
entity extraction, language detection + translation, LLM (prompts + Gemini + offline fallback),
file extraction (TXT / PDF / CSV).
"""
import csv as csv_mod
import inspect, io, json, re, random
import numpy as np, requests
from functools import lru_cache
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pypdf import PdfReader

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"

import os
MAX_TEXT_CHARS = 2000
MAX_BATCH_ROWS = 100
MAX_FILE_BYTES = 2 * 1024 * 1024
_GEMINI_INVALID = False

def get_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()

def gemini_available() -> bool:
    return bool(get_api_key()) and not _GEMINI_INVALID

def get_gemini_models() -> list[str]:
    return [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest"] if m]

# ===================== MODEL LOADING =====================
@lru_cache(maxsize=1)
def _load_model_dict() -> dict:
    f = ARTIFACTS / "model.json"
    if not f.exists():
        raise RuntimeError("backend/artifacts/model.json not found - run NLP_Pipeline.ipynb and copy its artifacts")
    with open(f, "r", encoding="utf-8") as fp:
        return json.load(fp)

# Load stopwords exported by the notebook (sklearn stopwords minus 'not', 'no', 'never')
STOP = set(_load_model_dict()["stopwords"])

# ===================== TEXT CLEANING =====================
def stem(w: str) -> str:
    """Very small suffix stemmer — just enough to group inflections
    (e.g. 'requests' -> 'request', 'ragging' -> 'ragg')."""
    if len(w) > 5 and w.endswith("ing"): return w[:-3]
    if len(w) > 4 and w.endswith("ed"):  return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"): return w[:-1]
    return w

def clean_steps(text: str) -> dict:
    """Return every intermediate step of our preprocessing pipeline.
    Used by the /api/steps endpoint to show the examiner exactly what happens."""
    lower = text.lower()
    no_punct = " ".join(re.sub(r"[^a-z0-9\s]", " ", lower).split())
    tokens = no_punct.split()
    no_stop = [t for t in tokens if t not in STOP and len(t) > 1 and not t.isdigit()]
    stemmed = [stem(t) for t in no_stop]
    return {"original": text, "lowercase": lower, "no_punctuation": no_punct,
            "tokens": tokens, "no_stopwords": no_stop, "stemmed": stemmed,
            "cleaned": " ".join(stemmed)}

def clean(text: str) -> str:
    """Full preprocessing in one call: lowercase, strip punctuation,
    remove stopwords (keeping negations), light stemming."""
    return clean_steps(text)["cleaned"]

# ===================== PREDICTION WITHOUT SCIKIT-LEARN =====================
class PortableModel:
    """Exactly what the web app does at prediction time (NumPy only)."""
    def __init__(self, d):
        self.vocab = d["tfidf"]["vocabulary"]; self.idf = np.array(d["tfidf"]["idf"])
        self.heads = {k: (d[k]["classes"], np.array(d[k]["coef"]), np.array(d[k]["intercept"])) for k in ("category", "priority")}
    def vectorize(self, cleaned):
        toks = re.findall(r"(?u)\b\w\w+\b", cleaned)
        counts = {}
        for g in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]:
            i = self.vocab.get(g)
            if i is not None: counts[i] = counts.get(i, 0) + 1
        if not counts: return np.array([], dtype=int), np.array([])
        idx = np.array(list(counts)); x = (1 + np.log(np.array([counts[i] for i in idx], float))) * self.idf[idx]
        return idx, x / np.linalg.norm(x)
    def proba(self, cleaned, target):
        classes, coef, inter = self.heads[target]; idx, x = self.vectorize(cleaned)
        s = coef[:, idx] @ x + inter; e = np.exp(s - s.max()); return dict(zip(classes, e / e.sum()))

@lru_cache(maxsize=1)
def get_model() -> PortableModel:
    """Load model once at startup."""
    return PortableModel(_load_model_dict())

def get_top_tfidf_terms(cleaned: str, top_n: int = 8) -> list[dict]:
    """Return top terms by TF-IDF weight for a cleaned text using PortableModel."""
    model = get_model()
    idx, x = model.vectorize(cleaned)
    if len(idx) == 0:
        return []
    # Build inverse vocabulary lookup (index -> term)
    inv_vocab = {i: term for term, i in model.vocab.items()}
    ranked = sorted(zip(idx, x), key=lambda item: item[1], reverse=True)[:top_n]
    return [{"term": inv_vocab[i], "score": round(float(weight), 3)} for i, weight in ranked]

_PRIORITY_RULES = {
    "Critical": [
        ("harassment", r"harass\w*"), ("ragging", r"ragging"), ("assault", r"assault\w*"),
        ("threat", r"threat\w*"), ("suicide", r"suicid\w*"), ("self harm", r"self[\s-]+harm"),
        ("fire", r"fire"), ("electric shock", r"electric\s+shock"),
        ("medical emergency", r"medical\s+emergency"), ("ambulance", r"ambulance"),
        ("unsafe", r"unsafe"), ("stalking", r"stalk\w*"),
        ("discrimination", r"discriminat\w*"),
    ],
    "High": [
        ("no water for days", r"no\s+(?:hot\s+)?water\s+for\s+(?:days|\w+\s+days)"), ("no heating", r"no\s+heating"),
        ("deadline tomorrow", r"deadline\s+tomorrow"), ("exam tomorrow", r"exam\s+tomorrow"),
        ("fee deadline", r"fee\s+deadline"), ("scholarship not received", r"scholarship\s+not\s+received"),
        ("locked out", r"locked\s+out"), ("theft", r"theft"), ("stolen", r"stolen"),
        ("infestation", r"infestation"), ("mold", r"mold"), ("safety hazard", r"safety\s+hazard"),
    ],
}
_PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

def _priority_rules(text: str, model_priority: str) -> tuple[str, str, list[dict]]:
    matches = []
    for priority, rules in _PRIORITY_RULES.items():
        for label, pattern in rules:
            if re.search(rf"\b(?:{pattern})\b", text.lower()):
                matches.append({"term": label, "priority": priority})
    best_rule = min((m["priority"] for m in matches), key=_PRIORITY_ORDER.get, default=None)
    if best_rule and _PRIORITY_ORDER[best_rule] < _PRIORITY_ORDER[model_priority]:
        return best_rule, "rule", matches
    return model_priority, "model", matches

def _explain(cleaned: str, target: str, label: str, top_n: int = 5) -> list[dict]:
    model = get_model()
    classes, coef, _ = model.heads[target]
    idx, values = model.vectorize(cleaned)
    if len(idx) == 0:
        return []
    class_index = classes.index(label)
    inv_vocab = {i: term for term, i in model.vocab.items()}
    contributions = values * coef[class_index, idx]
    ranked = sorted(zip(idx, contributions), key=lambda pair: abs(float(pair[1])), reverse=True)[:top_n]
    return [{"term": inv_vocab[i], "weight": round(float(weight), 4)} for i, weight in ranked]

def classify_detailed(text: str) -> dict:
    """Classify with model scores, compact feature contributions and domain rules."""
    model = get_model()
    cleaned = clean(text)
    category_probs = model.proba(cleaned, "category")
    priority_probs = model.proba(cleaned, "priority")
    cat = max(category_probs, key=category_probs.get)
    model_priority = max(priority_probs, key=priority_probs.get)
    pri, source, matched = _priority_rules(text, model_priority)
    confidence = float(category_probs[cat])
    top_categories = sorted(category_probs.items(), key=lambda pair: pair[1], reverse=True)[:3]
    return {
        "category": cat,
        "priority": pri,
        "confidence": round(confidence, 3),
        "category_probs": {k: round(float(v), 3) for k, v in sorted(category_probs.items())},
        "priority_probs": {k: round(float(v), 3) for k, v in sorted(priority_probs.items())},
        "priority_source": source,
        "matched_rules": matched,
        "explanation": {"category_terms": _explain(cleaned, "category", cat),
                        "priority_terms": _explain(cleaned, "priority", model_priority)},
        "needs_review": confidence < 0.5,
        "top_categories": [{"category": name, "probability": round(float(prob), 3)}
                           for name, prob in top_categories] if confidence < 0.5 else [],
        "model_priority": model_priority,
    }

def classify(text: str) -> dict:
    """Predict category and final priority using PortableModel and safety rules."""
    result = classify_detailed(text)
    result.pop("model_priority")
    return result

# ===================== SENTIMENT =====================
_VADER = SentimentIntensityAnalyzer()
_VADER.lexicon.update({"please": 0})

def sentiment(text: str) -> tuple[str, float]:
    """Return (label, compound_score). Thresholds: <=-0.05 Negative, >=0.05 Positive."""
    s = _VADER.polarity_scores(text)["compound"]
    label = "Negative" if s <= -0.05 else "Positive" if s >= 0.05 else "Neutral"
    return label, round(s, 3)

# ===================== ENTITY EXTRACTION =====================
_REGEX_RULES = [
    ("COURSE", re.compile(r"\b[A-Z]{2,5}\s+\d{3}[A-Z]?\b")),
    ("BUILDING", re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+(?:Hall|Library|Building|Center|Centre|Lab|Campus)\b")),
    ("TIME", re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\b")),
    ("DAY", re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|yesterday|today|tomorrow|Yesterday|Today|Tomorrow)\b")),
]

def get_ner():
    return None, "domain regex rules (memory-optimised alternative to spaCy)"

def extract_entities(text: str) -> list[dict]:
    matches = []
    for label, pattern in _REGEX_RULES:
        for m in pattern.finditer(text):
            matches.append({"text": m.group(0), "label": label, "start": m.start(), "end": m.end()})

    matches.sort(key=lambda x: (x["start"], -x["end"]))
    filtered = []
    last_end = -1
    for ent in matches:
        if ent["start"] >= last_end:
            filtered.append(ent)
            last_end = ent["end"]
    return filtered

# ===================== LANGUAGE DETECTION + TRANSLATION =====================
def detect_language(text: str) -> str:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    try:
        return detect(text)
    except Exception:
        return "en"

_TRANS_CACHE: dict = {}

def to_english(text: str) -> tuple[str, str, str | None]:
    """Translate text to English. Returns (english_text, language_code, service_used)."""
    lang = detect_language(text)
    if lang == "en":
        return text, "en", None
    if text in _TRANS_CACHE:
        return _TRANS_CACHE[text], lang, "cache"

    def ok(t, s):
        _TRANS_CACHE[text] = t
        return t, lang, s

    if get_api_key():
        try:
            t, m = call_gemini(f"Translate to English. Output ONLY the translation:\n{text}", timeout=15)
            return ok(t.strip(), f"Gemini ({m})")
        except Exception:
            pass
    try:
        from deep_translator import GoogleTranslator
        return ok(GoogleTranslator(source="auto", target="en").translate(text), "Google Translate")
    except Exception:
        pass
    try:
        from deep_translator import MyMemoryTranslator
        src = "mr-IN" if lang == "mr" else "hi-IN"
        return ok(MyMemoryTranslator(source=src, target="en-GB").translate(text), "MyMemory")
    except Exception:
        return text, lang, "unavailable (original text used)"

# ===================== LLM: PROMPTS + GEMINI + OFFLINE FALLBACK =====================
DEPT = {
    "IT Support": "IT Helpdesk",
    "Academic": "Academic Office / Department",
    "Facilities": "Estates & Maintenance",
    "Housing": "Housing Office / Hostel Warden",
    "Campus Life": "Student Affairs",
    "Financial Aid": "Financial Aid Office",
    "Dining Services": "Dining Services Manager",
}
SLA = {
    "Critical": "within 24 hours",
    "High": "within 48 hours",
    "Medium": "within 5 working days",
    "Low": "within 10 working days",
}

PROMPT_V1 = 'Write a reply to this student grievance: "{g}"'

PROMPT_V2 = """You are a university grievance-redressal assistant.
Student grievance: \"\"\"{g}\"\"\"
Our system's analysis -> category: {cat}; priority: {pri}; sentiment: {sen}; entities found: {ents}.
Return ONLY valid JSON with exactly these keys:
{{"summary": "max 20 words", "department": "office that should handle it", "action": "one concrete next step", "reply": "polite reply, exactly 3 sentences"}}
Rules: be empathetic; do not promise outcomes; do not invent facts that are not in the grievance; mention the response time '{sla}'."""

def call_gemini(prompt: str, as_json: bool = False, timeout: int = 25) -> tuple[str, str]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY configured")
    cfg = {"responseMimeType": "application/json"} if as_json else {}
    last = ""
    for m in dict.fromkeys(get_gemini_models()):
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                headers={"x-goog-api-key": api_key},
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg},
                timeout=timeout,
            )
            if response.status_code in (401, 403):
                global _GEMINI_INVALID
                _GEMINI_INVALID = True
                raise RuntimeError("Gemini key is invalid or lacks permission.")
            if response.status_code == 429:
                raise RuntimeError("Gemini is busy right now. Please try again later.")
            response.raise_for_status()
            r = response.json()
            if "candidates" in r:
                return r["candidates"][0]["content"]["parts"][0]["text"], m
            last = f"{m}: {r.get('error', {}).get('message', str(r))[:150]}"
        except RuntimeError:
            raise
        except Exception as e:
            last = f"{m}: {type(e).__name__}"
    raise RuntimeError(last if last and "HTTPError" not in last else "Gemini is temporarily unavailable. Please try again later.")

def offline_reply(cat: str, pri: str) -> dict:
    d = DEPT.get(cat, "Student Section")
    return {
        "summary": f"{cat} grievance ({pri} priority).",
        "department": d,
        "action": f"Forward to {d} and track resolution {SLA.get(pri, 'within 10 working days')}.",
        "reply": f"Thank you for raising this concern. We have forwarded it to the {d}. You can expect a response {SLA.get(pri, 'within 10 working days')}.",
        "source": "offline template",
    }

def build_prompt(text: str, cat: str, pri: str, sen: str, ents: list[dict]) -> str:
    return PROMPT_V2.format(g=text, cat=cat, pri=pri, sen=sen,
                            ents=[e["text"] for e in ents], sla=SLA.get(pri, "within 10 working days"))

def make_reply(text: str, cat: str, pri: str, sen: str, ents: list[dict]) -> dict:
    if not get_api_key():
        return {**offline_reply(cat, pri), "note": "AI assistant is not configured; showing a standard template."}
    if not gemini_available():
        return {**offline_reply(cat, pri), "note": "AI assistant is unavailable; showing a standard template."}
    try:
        out, model = call_gemini(build_prompt(text, cat, pri, sen, ents), as_json=True)
        r = json.loads(out)
        if not all(k in r for k in ("summary", "department", "action", "reply")):
            raise ValueError("incomplete JSON")
        return {**r, "source": model}
    except Exception:
        return {**offline_reply(cat, pri), "note": "AI assistant is unavailable right now; showing a standard template."}

# ===================== FILE EXTRACTION =====================
class FileError(ValueError):
    pass

def _letter_body(text: str) -> str:
    """Extract the grievance body from a formal letter.
    Uses (?:\\n\\s*Regards|\\Z) so single-newline PDF extractions are handled correctly."""
    m = re.search(r"Grievance:\s*(.+?)(?:\n\s*Regards|\Z)", text, flags=re.S)
    return " ".join((m.group(1) if m else text).split())

def extract_file(name: str, raw: bytes) -> dict:
    if len(raw) > MAX_FILE_BYTES:
        raise FileError("File is too large (max 2 MB).")
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if ext == "csv":
        try:
            text_data = raw.decode("utf-8", errors="ignore")
            reader = csv_mod.DictReader(io.StringIO(text_data))
            if reader.fieldnames is None or "text" not in reader.fieldnames:
                raise FileError("The CSV needs a column named 'text'.")
            texts = [" ".join(row["text"].split()) for row in reader
                     if row.get("text") and row["text"].strip()]
        except FileError:
            raise
        except Exception:
            raise FileError("Could not read this CSV file.")
        if not texts:
            raise FileError("The 'text' column is empty.")
        return {"kind": "table", "name": name, "texts": texts[:MAX_BATCH_ROWS],
                "total_rows": len(texts), "truncated": len(texts) > MAX_BATCH_ROWS}

    if ext == "pdf":
        try:
            reader = PdfReader(io.BytesIO(raw))
            if len(reader.pages) > 10:
                raise FileError("PDF is too long (max 10 pages).")
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except FileError:
            raise
        except Exception:
            raise FileError("Could not read this PDF.")
    elif ext == "txt":
        text = raw.decode("utf-8", errors="ignore")
    else:
        raise FileError("Unsupported file type. Use TXT, PDF or CSV.")

    body = _letter_body(text)
    if len(body) < 5:
        raise FileError("No readable text found in the file.")
    return {"kind": "text", "name": name, "text": body[:MAX_TEXT_CHARS]}

# ===================== FULL PIPELINE =====================
def analyze(text: str, with_reply: bool = True) -> dict:
    english, lang, svc = to_english(text)
    cls = classify_detailed(english)
    sen, score = sentiment(english)
    ents = extract_entities(english)
    reply = make_reply(english, cls["category"], cls["priority"], sen, ents) if with_reply else None

    # Warning for non-English texts when translation was unavailable
    warning = None
    if lang != "en" and svc == "unavailable (original text used)":
        warning = "Translation was unavailable. Classification on non-English text may be unreliable."

    return {
        "text": text, "english_text": english,
        "language": lang, "translated_by": svc,
        "category": cls["category"], "priority": cls["priority"],
        "confidence": cls["confidence"],
        "explanation": cls["explanation"], "priority_source": cls["priority_source"],
        "matched_rules": cls["matched_rules"], "needs_review": cls["needs_review"],
        "top_categories": cls["top_categories"],
        "sentiment": sen, "sentiment_score": score,
        "entities": ents, "reply": reply,
        "warning": warning,
    }

def find_similar(english_text: str, rows: list[dict], exclude_id: int | None = None) -> list[dict]:
    """Find the closest stored grievances with the exported TF-IDF features."""
    model = get_model()
    query_idx, query_values = model.vectorize(clean(english_text))
    if len(query_idx) == 0:
        return []
    query = dict(zip(query_idx.tolist(), query_values.tolist()))
    matches = []
    for row in rows:
        if row["id"] == exclude_id:
            continue
        idx, values = model.vectorize(clean(row["english_text"]))
        similarity = sum(query.get(int(i), 0.0) * float(v) for i, v in zip(idx, values))
        if similarity >= 0.35:
            matches.append({"id": row["id"], "english_text": row["english_text"][:140],
                            "category": row["category"], "status": row["status"],
                            "similarity": round(float(similarity), 3)})
    return sorted(matches, key=lambda row: row["similarity"], reverse=True)[:3]

# ===================== HELPOUT ROWS FOR WALKTHROUGH =====================
def get_held_out_rows(n: int = 20) -> list[dict]:
    """Read backend/artifacts/heldout.csv using the csv module."""
    f = ARTIFACTS / "heldout.csv"
    if not f.exists():
        raise RuntimeError("backend/artifacts/heldout.csv not found")
    with open(f, encoding="utf-8") as fp:
        rows = list(csv_mod.DictReader(fp))
    return random.sample(rows, min(n, len(rows)))

# Functions exposed to /api/source for the Walkthrough "Show code" feature
SOURCE_WHITELIST = {
    "extract_file": extract_file,
    "clean_steps": clean_steps,
    "classify": classify,
    "PortableModel": PortableModel,
    "sentiment": sentiment,
    "extract_entities": extract_entities,
    "to_english": to_english,
    "build_prompt": build_prompt,
    "make_reply": make_reply,
}

def get_source(name: str) -> str | None:
    fn = SOURCE_WHITELIST.get(name)
    if fn is None:
        return None
    return inspect.getsource(fn)
