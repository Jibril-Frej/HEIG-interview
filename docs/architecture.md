# Architecture Reference

> Load this file when working on system design, data flow, or deployment questions.
> Reference in Claude Code with: @docs/architecture.md

## Data flow

```
[Public: opentransportdata.swiss]          [Private: synthetic]
  Occupancy JSON (daily, 3 months)           Employee records (CSV)
  IstDaten CSV (historical delays)           - home_station
  GTFS (static timetable)                    - work_station
           │                                 - usual_departure_time
           │                                 - remote_work_days
           ▼                                 - travel_class
    data/download_sbb.py              data/generate_synthetic.py
           │                                 │
           └──────────────┬──────────────────┘
                          ▼
                  data/build_features.py
                          │
                          ▼
                  dataset.parquet
                  (one row per train-section-date)
                  features: hour, dow, holiday, delay_avg,
                            n_cars, corporate_load, ...
                  label: occupancy_2nd ∈ {0=low,1=med,2=high}
                         occupancy_1st ∈ {0=low,1=med,2=high}
```

## Model

```
Input:  (batch, seq_len=14, n_features)   # 14 days history
GRU:    2 layers, hidden_size=64
Head:   Linear(64, 3) × 2                # one per class
Loss:   CrossEntropyLoss (per head)
Output: occupancy probabilities + argmax label
```

Saved artifacts:
- `model/artifacts/gru_model.pt`
- `model/artifacts/scaler.pkl`
- `model/artifacts/feature_columns.json`

## API (FastAPI on Cloud Run)

```
POST /predict
  body: { line, date, departure_station, travel_class }
  → { occupancy_label, confidence, probabilities }

POST /chat
  body: { message }
  → { reply, trips: [...] }   # trips contains predict results
```

Chat logic (chat.py):
1. Regex parse: extract origin, destination, time from message
2. Look up next 3 departures from GTFS timetable
3. Call /predict for each
4. Format reply: "The 07:12 Basel→Zürich will be 🔴 busy (85% confidence)"

## Frontend (Streamlit on Cloud Run)

- Single page: chat history + text input
- Calls POST /chat on submit
- Renders occupancy as emoji: 🟢 low / 🟡 medium / 🔴 high
- Shows confidence percentage

## GCP Layout

```
Cloud Run: sbb-api          (api/ Docker image)
Cloud Run: sbb-frontend     (frontend/ Docker image)
Cloud Storage: gs://<bucket>/
    ├── data/dataset.parquet
    ├── model/gru_model.pt
    ├── model/scaler.pkl
    └── model/feature_columns.json
Artifact Registry: <region>-docker.pkg.dev/<project>/sbb/
    ├── sbb-api:latest
    └── sbb-frontend:latest
```

## Environment variables (set in Cloud Run)

| Service  | Variable         | Value                          |
|----------|-----------------|--------------------------------|
| api      | GCS_BUCKET       | your-bucket-name               |
| api      | MODEL_PATH       | model/gru_model.pt             |
| frontend | API_URL          | https://sbb-api-<hash>.run.app |
