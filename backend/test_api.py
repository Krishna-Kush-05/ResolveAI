"""Tests for the ResolveAI backend.
Covers: analyse, bad inputs, TXT/PDF/CSV extraction, path traversal,
batch, history CRUD, source whitelist, steps, metrics, and multilingual.
Run: cd backend && python -m pytest test_api.py -v
"""
import csv, os, sys, tempfile
from pathlib import Path

# Point to a temp DB so tests don't touch any real data
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ["GEMINI_API_KEY"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from main import app
import nlp


def test_full_flow():
    with TestClient(app) as c:
        # 1. Health check
        h = c.get("/api/health").json()
        assert h["status"] == "ok"
        assert h["llm_configured"] is False
        assert "regex rules" in h["ner"]

        # 2. Analyse a valid grievance (English)
        r = c.post("/api/analyze", json={"text": "There has been no hot water in Cedar Hall for three days."}).json()
        assert r["category"] in ("Housing", "Facilities")
        assert r["priority"] in ("Low", "Medium", "High", "Critical")
        assert r["reply"]["source"] == "offline template"
        assert r["id"] is not None
        assert len(r["entities"]) >= 1  # Cedar Hall should be recognized as BUILDING
        assert any(e["label"] == "BUILDING" for e in r["entities"])

        # 3. Bad inputs validation
        assert c.post("/api/analyze", json={"text": "hi"}).status_code == 422           # too short (< 5)
        assert c.post("/api/analyze", json={"text": ""}).status_code == 422             # empty
        assert c.post("/api/analyze", json={"text": "x" * 2001}).status_code == 422     # too long (> 2000)

        # 4. File extraction validations
        assert c.post("/api/extract", files={"file": ("malicious.exe", b"binary content")}).status_code == 422
        assert c.post("/api/extract", files={"file": ("no_text.csv", b"a,b\n1,2")}).status_code == 422
        # File larger than 2 MB
        big_content = b"text\n" + (b"hello world\n" * 200000)
        assert c.post("/api/extract", files={"file": ("big.txt", big_content)}).status_code == 422

        # 5. Samples list and extraction
        s = c.get("/api/samples").json()
        assert len(s) >= 3

        # TXT sample
        t = c.get("/api/samples/letter_1.txt").json()
        assert t["kind"] == "text" and len(t["text"]) > 5

        # PDF sample
        pdfs = [x["name"] for x in s if x["name"].endswith(".pdf")]
        if pdfs:
            pdf_res = c.get(f"/api/samples/{pdfs[0]}").json()
            assert pdf_res["kind"] == "text"
            assert len(pdf_res["text"]) > 5

        # Path traversal protection
        assert c.get("/api/samples/..%2F..%2Fmain.py").status_code == 404
        assert c.get("/api/samples/%2E%2E%2Fmain.py").status_code == 404

        # 6. Batch CSV extraction and batch analysis
        b = c.get("/api/samples/grievances_batch.csv").json()
        assert b["kind"] == "table"
        assert len(b["texts"]) > 0
        batch_slice = b["texts"][:5]
        out = c.post("/api/analyze/batch", json={"texts": batch_slice}).json()
        assert out["count"] == 5
        assert "by_priority" in out
        assert "by_category" in out
        assert len(out["results"]) == 5

        # Batch validation
        assert c.post("/api/analyze/batch", json={"texts": []}).status_code == 422
        assert c.post("/api/analyze/batch", json={"texts": ["valid"] * 101}).status_code == 422

        # 7. History CRUD
        hist = c.get("/api/history").json()
        assert len(hist) == 6  # 1 single + 5 batch
        first_id = hist[0]["id"]
        assert c.delete(f"/api/history/{first_id}").status_code == 200
        # Delete non-existent ID -> 404
        assert c.delete("/api/history/999999").status_code == 404
        # Clear all
        assert c.delete("/api/history").json()["deleted"] == 5
        assert c.get("/api/history").json() == []

        # 8. Metrics endpoint
        m = c.get("/api/metrics").json()
        assert m["dataset"]["rows"] == 3000
        assert m["category"]["test"]["accuracy"] > 0.90
        assert m["priority"]["test"]["accuracy"] > 0.50
        assert "ner_counts" in m
        assert "topics" in m

        # 9. Source code whitelist endpoint
        for fn_name in ["clean_steps", "classify", "sentiment", "extract_entities", "to_english", "build_prompt", "make_reply", "extract_file"]:
            resp = c.get(f"/api/source/{fn_name}")
            assert resp.status_code == 200
            assert f"def {fn_name}" in resp.json()["source"]

        assert c.get("/api/source/nonexistent_func").status_code == 404
        assert c.get("/api/source/os").status_code == 404

        # 10. Pipeline Steps endpoint
        st = c.get("/api/steps", params={"text": "The Wi-Fi in Cedar Hall is broken since Monday."}).json()
        assert "clean_steps" in st
        assert "top_tfidf_terms" in st
        assert "category_probs" in st
        assert "priority_probs" in st
        assert st["sentiment"] in ("Negative", "Neutral", "Positive")
        assert len(st["entities"]) >= 1

        # 11. Sentiment check endpoint
        sc = c.get("/api/sentiment-check", params={"n": 5}).json()
        assert len(sc) == 5
        assert all("vader_label" in x and "vader_score" in x for x in sc)

        # 12. Prompt compare endpoint (without key -> 400 with helpful message)
        pc_err = c.post("/api/prompt-compare")
        assert pc_err.status_code == 400
        assert "GEMINI_API_KEY" in pc_err.json()["detail"]


def test_multilingual_fallback():
    # Hindi text translation fallback test
    hindi = "मेरे हॉस्टल में तीन दिन से गर्म पानी नहीं आ रहा है।"
    eng, lang, svc = nlp.to_english(hindi)
    assert lang == "hi"
    assert len(eng) > 5


def test_entity_patterns():
    text = "Students in CHEM 102 should report to Cedar Hall at 9:30 AM this Friday."
    ents = nlp.extract_entities(text)
    labels = {e["label"] for e in ents}
    assert "COURSE" in labels
    assert "BUILDING" in labels
    assert "TIME" in labels
    assert "DAY" in labels


def test_heldout_predictions_match_notebook_metrics():
    with open(nlp.ARTIFACTS / "heldout.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    metrics = __import__("json").loads((nlp.ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    predictions = [nlp.classify_detailed(r["complaint_text"]) for r in rows]
    category_correct = sum(p["category"] == r["category"] for p, r in zip(predictions, rows))
    model_priority_correct = sum(p["model_priority"] == r["urgency"] for p, r in zip(predictions, rows))
    hybrid_priority_correct = sum(p["priority"] == r["urgency"] for p, r in zip(predictions, rows))
    category_accuracy = category_correct / len(rows)
    model_priority_accuracy = model_priority_correct / len(rows)
    hybrid_priority_accuracy = hybrid_priority_correct / len(rows)
    print(f"Held-out category accuracy: {category_accuracy:.4f} (metrics: {metrics['category']['test']['accuracy']:.4f})")
    print(f"Held-out priority accuracy model-only: {model_priority_accuracy:.4f}")
    print(f"Held-out priority accuracy hybrid: {hybrid_priority_accuracy:.4f}")
    assert abs(category_accuracy - metrics["category"]["test"]["accuracy"]) <= 0.01
    assert abs(model_priority_accuracy - metrics["priority"]["test"]["accuracy"]) <= 0.01


def test_steps_and_pdf_letter_extraction():
    with TestClient(app) as c:
        response = c.get("/api/steps", params={"text": "The Wi-Fi in Cedar Hall is broken since Monday."})
        assert response.status_code == 200
        assert len(response.json()["top_tfidf_terms"]) <= 8

    pdf = nlp.SAMPLES_DIR / "letter_6.pdf"
    extracted = nlp.extract_file(pdf.name, pdf.read_bytes())["text"]
    assert extracted == "Need information about financial aid options for graduate students."
    assert "Regards" not in extracted


def test_history_status_updates_and_filter():
    with TestClient(app) as c:
        first = c.post("/api/analyze", json={"text": "No hot water in Cedar Hall for three days."}).json()
        second = c.post("/api/analyze", json={"text": "The campus Wi-Fi keeps disconnecting in the library."}).json()
        assert first["status"] == second["status"] == "pending"

        resolved = c.patch(f"/api/history/{first['id']}", json={"status": "resolved"})
        assert resolved.status_code == 200
        assert c.get("/api/history", params={"status": "resolved"}).json()[0]["id"] == first["id"]
        assert [row["id"] for row in c.get("/api/history", params={"status": "pending"}).json()] == [second["id"]]

        bulk = c.patch("/api/history", json={"ids": [first["id"], second["id"]], "status": "resolved"})
        assert bulk.status_code == 200 and bulk.json()["updated"] == 2
        assert len(c.get("/api/history", params={"status": "resolved"}).json()) == 2

        assert c.patch(f"/api/history/{second['id']}", json={"status": "done"}).status_code == 422
        assert c.patch("/api/history/999999", json={"status": "resolved"}).status_code == 404


def test_existing_database_status_migration(tmp_path, monkeypatch):
    import sqlite3
    old_db = tmp_path / "old.db"
    with sqlite3.connect(old_db) as conn:
        conn.execute("""CREATE TABLE grievances (
            id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, english_text TEXT NOT NULL,
            language TEXT NOT NULL, category TEXT NOT NULL, priority TEXT NOT NULL,
            confidence REAL NOT NULL, sentiment TEXT NOT NULL, sentiment_score REAL NOT NULL,
            entities TEXT NOT NULL DEFAULT '[]', reply TEXT, created_at TEXT NOT NULL)""")
        conn.execute("""INSERT INTO grievances
            (text, english_text, language, category, priority, confidence, sentiment, sentiment_score, created_at)
            VALUES ('existing grievance', 'existing grievance', 'en', 'Academic', 'Low', 0.5, 'Neutral', 0, 'today')""")
    monkeypatch.setattr(__import__("db"), "DB_PATH", str(old_db))
    __import__("db").init_db()
    rows = __import__("db").get_history()
    assert rows[0]["text"] == "existing grievance"
    assert rows[0]["status"] == "pending" and rows[0]["resolved_at"] is None


def test_similar_grievances_endpoint():
    text = "There has been no hot water in Cedar Hall for several days."
    with TestClient(app) as c:
        first = c.post("/api/analyze", json={"text": text}).json()
        second = c.post("/api/analyze", json={"text": text}).json()
        assert "explanation" in second and "priority_source" in second and "similar" in second
        assert "explanation" in second and "priority_source" in second and "similar" in second
        matches = c.get("/api/similar", params={"id": second["id"]})
        assert matches.status_code == 200
        assert matches.json()[0]["id"] == first["id"]
        assert matches.json()[0]["similarity"] >= 0.35
        assert c.get("/api/similar", params={"id": 999999}).status_code == 404


def test_stats_endpoint():
    with TestClient(app) as c:
        c.post("/api/analyze", json={"text": "The library Wi-Fi keeps disconnecting."})
        stats = c.get("/api/stats")
        assert stats.status_code == 200
        data = stats.json()
        assert data["total"] >= 1
        assert data["pending"] + data["resolved"] == data["total"]
        assert len(data["per_day"]) == 14
        assert "average_sentiment_score" in data and "by_language" in data


def test_explanation_and_priority_rule():
    result = nlp.classify_detailed("I am being harassed and feel unsafe.")
    assert result["priority"] == "Critical"
    assert any(rule["term"] == "harassment" for rule in result["matched_rules"])
    assert len(result["explanation"]["category_terms"]) <= 5
    assert len(result["explanation"]["priority_terms"]) <= 5
    assert nlp._priority_rules("a threat was made", "Low")[1] == "rule"
    assert nlp._priority_rules("I enjoy fireworks", "Low")[1] == "model"
    assert nlp._priority_rules("No water for three days", "Low")[0] == "High"
    assert nlp._priority_rules("No water for three days", "Low")[0] == "High"
