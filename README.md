# ResolveAI

AI-assisted student grievance analysis and resolution tracking.

ResolveAI provides a React frontend and FastAPI backend for classifying student grievances, estimating urgency, scoring sentiment, extracting entities, and drafting a response. The notebook trains and evaluates the models; the deployed backend only loads exported artifacts.

Live demo: Add your Render URL here after deploy

## Open in Colab

Open the ResolveAI notebook on Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Krishna-Kush-05/ResolveAI/blob/main/NLP_Pipeline.ipynb)

## Workflow

1. Open `NLP_Pipeline.ipynb` in Colab. Add a secret named `GEMINI_API_KEY` to enable prompt comparison, then choose **Run all**.
2. The notebook downloads `artifacts.zip` when it finishes.
3. Copy `model.json`, `metrics.json`, and `heldout.csv` into `backend/artifacts/`, then commit and push. Render redeploys from the updated repository.

The model is exported as plain numbers (vocabulary, IDF weights, and classifier coefficients). This avoids scikit-learn version mismatches between Colab and Render and reduces runtime memory use.

## Project structure

```text
ResolveAI/
├── NLP_Pipeline.ipynb
├── data/complaints.csv
├── backend/
│   ├── main.py, nlp.py, db.py, test_api.py, requirements.txt
│   ├── artifacts/model.json, metrics.json, heldout.csv
│   └── data/samples/
├── frontend/src/
├── Dockerfile, .dockerignore, render.yaml
└── README.md
```

## Local setup (Windows PowerShell)

Prerequisites: Python 3.11 or 3.12 and Node.js 18 or newer.

```powershell
Set-Location backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest httpx
python -m pytest test_api.py -v
uvicorn main:app --reload --port 8000
```

In a second PowerShell window, build or run the frontend:

```powershell
Set-Location frontend
npm ci
npm run dev
```

For the production-style single server, run `npm run build` in `frontend`, then start `uvicorn main:app --port 8000` from `backend`. The backend serves `frontend/dist` when present.

## Render deployment

The multi-stage Dockerfile builds the React app and runs FastAPI. Connect the repository to Render using `render.yaml`; set `GEMINI_API_KEY` in Render if Gemini features are wanted. The backend does not train during deploy, so the three files in `backend/artifacts/` must be committed.

After adding the notebook artifacts and reviewing your changes, push from PowerShell:

```powershell
git add -A
git commit -m "Update ResolveAI branding and links"
git branch -M main
git remote add origin https://github.com/Krishna-Kush-05/ResolveAI.git
git push -u origin main
```

If `origin` is already configured, skip the `git remote add` line. Render redeploys after the push when the repository is connected.

## API key

The Gemini key is read from `GEMINI_API_KEY` at request time and is never returned to the browser. Restart the app after changing the environment. `GEMINI_MODEL` is optional.

- **Render:** Open the service's **Environment** tab, add or update `GEMINI_API_KEY`, then save and redeploy. Remove the variable to disable Gemini replies.
- **Local:** Put `GEMINI_API_KEY=your_key_here` in `backend/.env` (copy `.env.example` and replace the placeholder). To change or remove it, edit or delete that line and restart Uvicorn.
- **Docker:** Start with `docker run -e GEMINI_API_KEY=your_key_here ...`. Pass a new value on the next run to change it; omit the variable to remove it.
- **Railway, Fly.io, Heroku, and similar:** Add or change `GEMINI_API_KEY` in the app's environment/secrets settings and redeploy. Remove the variable to disable Gemini.

Never commit a real API key or include it in a frontend setting.

## Runtime dependencies

`backend/requirements.txt` pins FastAPI, Uvicorn, python-multipart, NumPy, vaderSentiment, langdetect, deep-translator, and pypdf; requests is also included. Training libraries are not required by the backend.
