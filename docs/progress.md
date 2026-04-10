# Project Progress

> Use this file to track what's done and what's next.
> Reference with @docs/progress.md — Claude Code can check off boxes as work completes.

## Phase 1 — Data
- [ ] `data/download_sbb.py` — fetch occupancy JSON for 1 operator (SBB) × 30 days
- [ ] `data/download_sbb.py` — sample IstDaten CSV (1 week, Zürich HB lines only)
- [ ] `data/generate_synthetic.py` — generate 500 employee records
- [ ] `data/build_features.py` — merge all sources → dataset.parquet
- [ ] Validate dataset: check shape, label distribution, nulls

## Phase 2 — Model
- [ ] `model/train.py` — data loader + train/val split
- [ ] `model/train.py` — GRU model definition
- [ ] `model/train.py` — training loop with loss logging
- [ ] `model/evaluate.py` — accuracy + confusion matrix
- [ ] Save model artifacts to `model/artifacts/`

## Phase 3 — API
- [ ] `api/main.py` — FastAPI skeleton with /health endpoint
- [ ] `api/predict.py` — load model from GCS + run inference
- [ ] `api/main.py` — POST /predict endpoint
- [ ] `api/chat.py` — regex intent parser
- [ ] `api/main.py` — POST /chat endpoint
- [ ] `api/Dockerfile` — build and test locally

## Phase 4 — Frontend
- [ ] `frontend/app.py` — Streamlit chat UI (static, no backend)
- [ ] `frontend/app.py` — connect to /chat endpoint
- [ ] `frontend/Dockerfile` — build and test locally

## Phase 5 — Deploy
- [ ] Push Docker images to Artifact Registry
- [ ] Deploy api to Cloud Run
- [ ] Deploy frontend to Cloud Run
- [ ] Set env vars + test end-to-end

## Notes / blockers
<!-- Add any issues or decisions here -->
