"""NLP core for ResolveAI.
Sections: cleaning, classification (PortableModel with NumPy), sentiment,
entity extraction, language detection + translation, LLM (prompts + Gemini + offline fallback),
file extraction (TXT / PDF / CSV).
"""
import csv as csv_mod
import inspect, io, json, re, random, time, logging
import numpy as np, requests
from urllib3.util import Timeout as RequestTimeout
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
_GEMINI_SKIP_UNTIL = 0.0
_GEMINI_LAST_ERROR: tuple[str, str] | None = None
log = logging.getLogger("resolveai.gemini")

def get_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()

def gemini_available() -> bool:
    return bool(get_api_key()) and not _GEMINI_INVALID and time.monotonic() >= _GEMINI_SKIP_UNTIL

def get_gemini_models() -> list[str]:
    return [os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"]

def gemini_diagnostics() -> dict:
    configured = bool(get_api_key())
    if not configured:
        state, message = "missing_key", "GEMINI_API_KEY is not set; using offline replies."
    elif _GEMINI_INVALID:
        state, message = "invalid_key", "Gemini rejected the API key; check GEMINI_API_KEY in Render."
    elif time.monotonic() < _GEMINI_SKIP_UNTIL:
        state, message = _GEMINI_LAST_ERROR or ("cooldown", "Gemini is temporarily unavailable; using offline replies.")
    else:
        state, message = "ready", "Gemini is configured and ready."
    return {"configured": configured, "ready": gemini_available(), "model": get_gemini_models()[0],
            "state": state, "message": message}

def log_gemini_configuration() -> None:
    status = gemini_diagnostics()
    log.info("Gemini startup: configured=%s model=%s transport=REST", status["configured"], status["model"])

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

def classify_detailed(text: str, include_explanation: bool = True) -> dict:
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
        "explanation": ({"category_terms": _explain(cleaned, "category", cat),
                         "priority_terms": _explain(cleaned, "priority", model_priority)}
                        if include_explanation else {"category_terms": [], "priority_terms": []}),
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

def to_english(text: str, allow_gemini: bool = True, detected_language: str | None = None) -> tuple[str, str, str | None]:
    """Translate text to English. Returns (english_text, language_code, service_used)."""
    lang = detected_language or detect_language(text)
    if lang == "en":
        return text, "en", None
    if text in _TRANS_CACHE:
        return _TRANS_CACHE[text], lang, "cache"

    def ok(t, s):
        _TRANS_CACHE[text] = t
        return t, lang, s

    if allow_gemini and get_api_key() and gemini_available():
        try:
            t, m = call_gemini(f"Translate to English. Output ONLY the translation:\n{text}", timeout=6)
            return ok(t.strip(), f"Gemini ({m})")
        except Exception:
            pass
    if not allow_gemini:
        return text, lang, "unavailable (original text used)"
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

def call_gemini(prompt: str, as_json: bool = False, timeout: int = 45) -> tuple[str, str]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY configured")
    cfg = {"responseMimeType": "application/json"} if as_json else {}
    global _GEMINI_SKIP_UNTIL, _GEMINI_LAST_ERROR
    model = get_gemini_models()[0]
    connect_timeout = min(5, timeout)
    read_timeout = timeout
    started = time.monotonic()
    log.info("event=gemini_called key_detected=true model=%s", model)
    log.info("event=gemini_request_started model=%s json=%s connect_timeout=%ss read_timeout=%ss total_timeout=%ss retries=0",
             model, as_json, connect_timeout, read_timeout, timeout)
    for m in [model]:
        try:
            request_timeout = RequestTimeout(total=timeout, connect=connect_timeout, read=read_timeout)
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                headers={"x-goog-api-key": api_key},
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg},
                timeout=request_timeout,
            )
            elapsed = time.monotonic() - started
            log.info("event=gemini_response_received model=%s http_status=%s response_header_elapsed_ms=%s total_elapsed_ms=%s",
                     m, response.status_code, round(response.elapsed.total_seconds() * 1000), round(elapsed * 1000))
            if response.status_code in (401, 403):
                global _GEMINI_INVALID
                detail = _google_error_message(response, api_key)
                if response.status_code == 401:
                    _GEMINI_INVALID = True
                    state, message = "invalid_key", "Gemini rejected GEMINI_API_KEY (401). Check the key in Render."
                else:
                    state, message = "permission_denied", "Gemini denied this API key (403). Check API access and key restrictions."
                _GEMINI_LAST_ERROR = (state, f"{message} {detail}".strip())
                _GEMINI_SKIP_UNTIL = time.monotonic() + 60
                log.error("event=gemini_api_error classification=%s status=%s model=%s elapsed_ms=%s detail=%s",
                          state, response.status_code, m, round(elapsed * 1000), detail)
                raise RuntimeError(_GEMINI_LAST_ERROR[1])
            if response.status_code == 429:
                detail = _google_error_message(response, api_key)
                message = "Gemini quota or rate limit reached (429). Check Google AI Studio usage and billing."
                _GEMINI_LAST_ERROR = ("quota", f"{message} {detail}".strip())
                _GEMINI_SKIP_UNTIL = time.monotonic() + 60
                log.warning("event=gemini_api_error classification=quota status=429 model=%s elapsed_ms=%s detail=%s",
                            m, round(elapsed * 1000), detail)
                raise RuntimeError(_GEMINI_LAST_ERROR[1])
            if response.status_code == 404:
                detail = _google_error_message(response, api_key)
                message = f"Gemini model '{m}' was not found (404). Check GEMINI_MODEL."
                _GEMINI_LAST_ERROR = ("unsupported_model", f"{message} {detail}".strip())
                _GEMINI_SKIP_UNTIL = time.monotonic() + 60
                log.error("event=gemini_api_error classification=unsupported_model status=404 model=%s elapsed_ms=%s detail=%s",
                          m, round(elapsed * 1000), detail)
                raise RuntimeError(_GEMINI_LAST_ERROR[1])
            if response.status_code >= 400:
                status_code = response.status_code
                category = "server_error" if status_code >= 500 else "request_error"
                detail = _google_error_message(response, api_key)
                message = f"Gemini request failed (HTTP {status_code})."
                _GEMINI_LAST_ERROR = (category, f"{message} {detail}".strip())
                _GEMINI_SKIP_UNTIL = time.monotonic() + 60
                log.error("event=gemini_api_error classification=%s status=%s model=%s elapsed_ms=%s detail=%s",
                          category, status_code, m, round(elapsed * 1000), detail)
                raise RuntimeError(_GEMINI_LAST_ERROR[1])
            response.raise_for_status()
            parse_started = time.monotonic()
            r = response.json()
            try:
                text = r["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                message = "Gemini returned no usable text (response may have been blocked)."
                _GEMINI_LAST_ERROR = ("empty_response", message)
                _GEMINI_SKIP_UNTIL = time.monotonic() + 60
                log.error("event=gemini_response_parse_error model=%s elapsed_ms=%s",
                          m, round((time.monotonic() - started) * 1000))
                raise RuntimeError(message)
            _GEMINI_LAST_ERROR = None
            log.info("event=gemini_response_parsed model=%s parse_ms=%s",
                     m, round((time.monotonic() - parse_started) * 1000))
            log.info("event=gemini_generation_completed model=%s total_elapsed_ms=%s",
                     m, round((time.monotonic() - started) * 1000))
            return text, m
        except requests.ConnectTimeout as e:
            message = "Timed out connecting to Gemini; check Render outbound DNS/network access."
            _GEMINI_LAST_ERROR = ("connect_timeout", message)
            _GEMINI_SKIP_UNTIL = time.monotonic() + 60
            log.warning("event=gemini_timeout classification=connect_timeout model=%s elapsed_ms=%s exception=%s",
                        m, round((time.monotonic() - started) * 1000), type(e).__name__)
            raise RuntimeError(message)
        except requests.ReadTimeout as e:
            message = "Gemini response timed out while waiting for generated content."
            _GEMINI_LAST_ERROR = ("read_timeout", message)
            _GEMINI_SKIP_UNTIL = time.monotonic() + 60
            log.warning("event=gemini_timeout classification=read_timeout model=%s elapsed_ms=%s exception=%s",
                        m, round((time.monotonic() - started) * 1000), type(e).__name__)
            raise RuntimeError(message)
        except requests.Timeout as e:
            message = "Gemini request exceeded its overall timeout."
            _GEMINI_LAST_ERROR = ("request_timeout", message)
            _GEMINI_SKIP_UNTIL = time.monotonic() + 60
            log.warning("event=gemini_timeout classification=request_timeout model=%s elapsed_ms=%s exception=%s",
                        m, round((time.monotonic() - started) * 1000), type(e).__name__)
            raise RuntimeError(message)
        except requests.RequestException as e:
            detail = str(e).replace(api_key, "[redacted]")[:300]
            detail = re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1[redacted]@", detail)
            lower_error = detail.lower()
            state = "dns_error" if any(s in lower_error for s in ("nameresolutionerror", "getaddrinfo", "name or service not known", "temporary failure in name resolution")) else "tls_error" if isinstance(e, requests.exceptions.SSLError) else "proxy_error" if isinstance(e, requests.exceptions.ProxyError) else "network_error"
            message = "Gemini DNS lookup failed." if state == "dns_error" else "Gemini TLS connection failed." if state == "tls_error" else "Gemini proxy connection failed." if state == "proxy_error" else "Gemini connection failed."
            _GEMINI_LAST_ERROR = (state, message)
            _GEMINI_SKIP_UNTIL = time.monotonic() + 60
            log.error("event=gemini_connection_error classification=%s model=%s elapsed_ms=%s exception=%s detail=%s",
                      state, m, round((time.monotonic() - started) * 1000), type(e).__name__, detail)
            raise RuntimeError(message)
        except RuntimeError:
            if _GEMINI_LAST_ERROR is None:
                _GEMINI_LAST_ERROR = ("request_error", "Gemini request failed; using offline replies.")
            _GEMINI_SKIP_UNTIL = time.monotonic() + 60
            raise
        except Exception as e:
            message = "Gemini response could not be parsed; using offline replies."
            _GEMINI_LAST_ERROR = ("response_error", message)
            _GEMINI_SKIP_UNTIL = time.monotonic() + 60
            log.exception("Gemini response handling failed: model=%s error=%s", m, type(e).__name__)
            raise RuntimeError(message) from e

def _google_error_message(response, api_key: str) -> str:
    try:
        message = response.json().get("error", {}).get("message", "")
        return str(message).replace(api_key, "[redacted]")[:400]
    except Exception:
        return ""

def run_gemini_diagnostic() -> dict:
    """Make one isolated, real GenerateContent call without retries or offline fallback."""
    api_key = get_api_key()
    model = get_gemini_models()[0]
    started = time.monotonic()
    log.info("event=gemini_diagnostic_started key_detected=%s model=%s", bool(api_key), model)
    if not api_key:
        return {"api_connectivity": "not_configured", "http_status": None, "model": model,
                "duration_ms": round((time.monotonic() - started) * 1000),
                "real_response_received": False, "response_text": None,
                "error_classification": "missing_key",
                "error": "GEMINI_API_KEY is not configured in this process."}

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key},
            json={"contents": [{"parts": [{"text": "Reply with exactly: GEMINI_CONNECTION_SUCCESS"}]}],
                  "generationConfig": {}},
            timeout=RequestTimeout(total=45, connect=5, read=45),
        )
    except requests.ConnectTimeout as e:
        return _diagnostic_failure(model, started, "connect_timeout", type(e).__name__, "Connection to Google timed out.")
    except requests.ReadTimeout as e:
        return _diagnostic_failure(model, started, "read_timeout", type(e).__name__, "Google did not return generated content before the read timeout.")
    except requests.Timeout as e:
        return _diagnostic_failure(model, started, "request_timeout", type(e).__name__, "The Gemini request exceeded its overall timeout.")
    except requests.exceptions.SSLError as e:
        return _diagnostic_failure(model, started, "tls_error", type(e).__name__, "TLS verification or handshake with Google failed.")
    except requests.exceptions.ProxyError as e:
        return _diagnostic_failure(model, started, "proxy_error", type(e).__name__, "The configured outbound proxy could not reach Google.")
    except requests.RequestException as e:
        detail = str(e).replace(api_key, "[redacted]")[:300]
        detail = re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1[redacted]@", detail)
        lower = detail.lower()
        category = "dns_error" if any(x in lower for x in ("nameresolutionerror", "getaddrinfo", "name or service not known", "temporary failure in name resolution")) else "connection_error"
        return _diagnostic_failure(model, started, category, type(e).__name__, detail or "Google connection failed.")

    elapsed_ms = round((time.monotonic() - started) * 1000)
    status = response.status_code
    log.info("event=gemini_diagnostic_http_response model=%s status=%s ttfb_ms=%s total_ms=%s",
             model, status, round(response.elapsed.total_seconds() * 1000), elapsed_ms)
    if status == 401:
        category, message = "invalid_key", "Google rejected the Gemini API key."
    elif status == 403:
        category, message = "permission_denied", "Google denied this key or its API permissions."
    elif status == 404:
        category, message = "model_not_found", f"Google could not find model '{model}'."
    elif status == 429:
        category, message = "quota_exceeded", "Google rate limit or quota was reached."
    elif status >= 500:
        category, message = "google_server_error", "Google returned a server error."
    elif status >= 400:
        category, message = "google_request_error", f"Google rejected the request with HTTP {status}."
    else:
        category, message = None, None
    if category:
        detail = _google_error_message(response, api_key)
        log.warning("event=gemini_diagnostic_failed classification=%s status=%s model=%s total_ms=%s detail=%s",
                    category, status, model, elapsed_ms, detail)
        return {"api_connectivity": "connected", "http_status": status, "model": model,
                "duration_ms": elapsed_ms, "real_response_received": False, "response_text": None,
                "error_classification": category, "error": f"{message} {detail}".strip()}
    try:
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (ValueError, KeyError, IndexError, TypeError):
        log.error("event=gemini_diagnostic_parse_failed status=%s model=%s total_ms=%s", status, model, elapsed_ms)
        return {"api_connectivity": "connected", "http_status": status, "model": model,
                "duration_ms": elapsed_ms, "real_response_received": False, "response_text": None,
                "error_classification": "invalid_response", "error": "Google returned HTTP success but no usable generated text."}
    log.info("event=gemini_diagnostic_succeeded model=%s status=%s total_ms=%s", model, status, elapsed_ms)
    return {"api_connectivity": "connected", "http_status": status, "model": model,
            "duration_ms": elapsed_ms, "real_response_received": True,
            "response_text": text[:200], "error_classification": None, "error": None}

def _diagnostic_failure(model: str, started: float, category: str, exception: str, message: str) -> dict:
    elapsed_ms = round((time.monotonic() - started) * 1000)
    log.warning("event=gemini_diagnostic_failed classification=%s model=%s total_ms=%s exception=%s",
                category, model, elapsed_ms, exception)
    return {"api_connectivity": "unreachable", "http_status": None, "model": model,
            "duration_ms": elapsed_ms, "real_response_received": False, "response_text": None,
            "error_classification": category, "error": message}

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
    global _GEMINI_LAST_ERROR, _GEMINI_SKIP_UNTIL
    reply_started = time.monotonic()
    if not get_api_key():
        log.warning("event=gemini_offline_fallback reason=missing_key elapsed_ms=%s", round((time.monotonic() - reply_started) * 1000))
        return {**offline_reply(cat, pri), "note": gemini_diagnostics()["message"]}
    if not gemini_available():
        diagnostic = gemini_diagnostics()
        log.warning("event=gemini_offline_fallback reason=%s elapsed_ms=%s",
                    diagnostic["state"], round((time.monotonic() - reply_started) * 1000))
        return {**offline_reply(cat, pri), "note": diagnostic["message"]}
    try:
        out, model = call_gemini(build_prompt(text, cat, pri, sen, ents), as_json=True)
        r = json.loads(out)
        if not all(k in r for k in ("summary", "department", "action", "reply")):
            raise ValueError("incomplete JSON")
        return {**r, "source": model}
    except Exception as e:
        if _GEMINI_LAST_ERROR:
            diagnostic = _GEMINI_LAST_ERROR[1]
        else:
            diagnostic = "Gemini returned a response that could not be parsed; using offline replies."
            _GEMINI_LAST_ERROR = ("invalid_response", diagnostic)
            _GEMINI_SKIP_UNTIL = time.monotonic() + 60
        log.warning("event=gemini_offline_fallback reason=%s elapsed_ms=%s exception=%s",
                    _GEMINI_LAST_ERROR[0] if _GEMINI_LAST_ERROR else "unknown",
                    round((time.monotonic() - reply_started) * 1000),
                    type(e).__name__)
        return {**offline_reply(cat, pri), "note": diagnostic}

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
def analyze(text: str, with_reply: bool = True, include_explanation: bool = True,
            translation: tuple[str, str, str | None] | None = None) -> dict:
    english, lang, svc = translation or to_english(text)
    cls = classify_detailed(english, include_explanation=include_explanation)
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

def find_similar(english_text: str, rows: list[dict], exclude_id: int | None = None, max_seconds: float | None = None) -> list[dict]:
    """Find the closest stored grievances with the exported TF-IDF features."""
    model = get_model()
    query_idx, query_values = model.vectorize(clean(english_text))
    if len(query_idx) == 0:
        return []
    query = dict(zip(query_idx.tolist(), query_values.tolist()))
    matches = []
    deadline = time.monotonic() + max_seconds if max_seconds is not None else None
    for row in rows:
        if deadline is not None and time.monotonic() >= deadline:
            break
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
