# SBB Transport Demand Forecasting — Project Context

## What this project is
A PoC combining public SBB open data with synthetic corporate mobility data
to predict train occupancy (low/medium/high) using a GRU model.
Deployed on GCP (Cloud Run) with a FastAPI backend and Streamlit chat frontend.

## Stack
- Python 3.11
- PyTorch (GRU model)
- FastAPI (API backend)
- Streamlit (chat frontend)
- GCP Cloud Run + Cloud Storage + Artifact Registry
- Docker (one image per service: api, frontend)

## Repo structure
```
project/
├── data/
│   ├── download_sbb.py        # fetch SBB occupancy JSON + IstDaten sample
│   ├── generate_synthetic.py  # generate corporate employee records
│   └── build_features.py      # merge & featurize → dataset.parquet
├── model/
│   ├── train.py               # GRU training loop
│   └── evaluate.py
├── api/
│   ├── main.py                # FastAPI app (/predict, /chat endpoints)
│   ├── predict.py             # model loading + inference
│   ├── chat.py                # intent parser + reply formatter
│   └── Dockerfile
├── frontend/
│   ├── app.py                 # Streamlit chat UI
│   └── Dockerfile
└── deploy/
    └── deploy.sh              # gcloud run deploy commands
```

## Workflow rules

### Before modifying any file
- State clearly what file you intend to change and why
- Wait for explicit confirmation ("yes", "ok", "go ahead") before proceeding
- Never modify multiple files at once without separate confirmations

### When making changes
- Make ONE small, focused change at a time
- Prefer editing 10-20 lines over rewriting whole files
- After each change, show a diff summary (what changed and why)

### Git commits
- Commit after every confirmed change
- Use this format: `<type>(<scope>): <short description>`
  - Types: feat, fix, data, model, deploy, docs, chore
  - Examples:
    - `feat(api): add /predict endpoint skeleton`
    - `data(sbb): add occupancy JSON downloader`
    - `model(gru): add 2-layer GRU training loop`
    - `deploy(cloudrun): add api Dockerfile`
- Never commit unconfirmed changes

### Code style
- Keep functions short (< 30 lines preferred)
- One function = one responsibility
- Add a one-line docstring to every function
- No unused imports

## Key data sources
- SBB Occupancy: https://data.opentransportdata.swiss/en/dataset/occupancy-forecast-json-dataset
- SBB IstDaten: https://opentransportdata.swiss/en/cookbook/actual-data/
- GTFS static: opentransportdata.swiss

## GCP config (fill in before deploying)
- PROJECT_ID=
- REGION=europe-west6  # Zurich
- BUCKET=
