# Breast Cancer Classification — ML Pipeline, API, and UI

## 1. Project Overview

This project builds an end-to-end machine learning workflow for classifying breast cancer tumors as **benign (B)** or **malignant (M)** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The repository covers:

- Data exploration and cleaning
- Preprocessing, feature selection, training, and evaluation
- A saved scikit-learn / imbalanced-learn model artifact
- A FastAPI prediction service
- A Streamlit user interface
- Docker packaging for both services
- Pytest checks for the model and API

The final model is an **imbalanced-learn pipeline** combining IQR outlier clipping, `RobustScaler`, `SMOTE`, and `LogisticRegression`, trained on **17 selected features**.

> This is an educational project. It is not a clinical diagnostic tool and must not be used for medical decisions.

## 2. Dataset

The project uses the WDBC dataset stored at `data/data.csv`.

| Property | Value |
|---|---:|
| Samples | 569 |
| Original input features | 30 numeric tumor measurements |
| Selected input features | 17 |
| Target | `diagnosis` |
| Classes | Benign (`B`), Malignant (`M`) |
| Class counts | 357 benign, 212 malignant |

The original columns also include `id` and an empty `Unnamed: 32` column. Both are removed during data preparation. The dataset is split into 80% training and 20% testing using stratification and `random_state=42`.

## 3. Project Evolution

### Task 01 — Baseline ML pipeline

- Explored the dataset and confirmed there were no duplicate rows or missing values after dropping `Unnamed: 32`.
- Encoded `diagnosis` as `B = 0` and `M = 1`.
- Split the data with `train_test_split`, `test_size=0.2`, `random_state=42`, and stratification.
- Applied IQR clipping using boundaries learned from the training set.
- Scaled features with `RobustScaler`.
- Used `SMOTE(random_state=42)` to address class imbalance during training.
- Trained a `LogisticRegression(max_iter=1000)` classifier.
- Evaluated accuracy, precision, recall, F1-score, and the confusion matrix.
- Saved the fitted pipeline to `models/model.joblib`.
- Moved the notebook workflow into reusable modules under `production_app/`.

The Task 01 baseline used all 30 original features.

### Task 02 — Feature selection

Three feature-selection approaches were considered:

| Approach | Description | Role in this project |
|---|---|---|
| Filter | Univariate scores, correlation, or statistical tests | Fast screening, but ignores interactions with the classifier |
| Wrapper | RFECV / recursive feature elimination using the classifier and cross-validation | **Selected approach** |
| Embedded | L1 regularization or tree-based feature importance | Model-specific; considered but not selected |

The final method is **RFECV (wrapper)** with `LogisticRegression`, 5-fold cross-validation, and accuracy scoring. RFECV retained **17 of the 30** features. The selected list is fixed in `production_app/config.py` so training and inference use the same schema.

The reduced feature set was then used by the production training pipeline, API schema, and Streamlit UI.

### Task 03 — Prediction service and delivery

- Added FastAPI endpoints:
  - `GET /health`
  - `POST /predict`
- Added Pydantic validation for the 17 selected features.
- Handled the case where the model artifact cannot be loaded: the API remains available and reports `503` from `/health`.
- Added a Streamlit UI that sends the selected features to the API and displays the prediction.
- Added `Dockerfile.api`, `Dockerfile.frontend`, and `docker-compose.yml`.
- Added model and API tests under `tests/`.

## 4. Feature Selection Results

### Approach comparison

- **Filter methods** are inexpensive and easy to interpret, but they evaluate each feature without considering how the final classifier combines features.
- **Wrapper methods** evaluate subsets using the actual estimator. RFECV repeatedly removes the least important feature and uses cross-validation to measure the resulting model. This is more expensive but aligns selection with the final model.
- **Embedded methods** such as L1-regularized models or tree importance combine selection with training, but the result is specific to that model family.

RFECV was chosen because it directly evaluates the LogisticRegression pipeline with 5-fold cross-validation and accuracy, while still producing a compact feature set.

### Selected input features

```text
texture_mean
perimeter_mean
area_mean
concavity_mean
concave points_mean
radius_se
area_se
compactness_se
fractal_dimension_se
radius_worst
texture_worst
perimeter_worst
area_worst
smoothness_worst
concavity_worst
concave points_worst
symmetry_worst
```

Two feature names contain spaces. The API accepts both the original names and their Python-style equivalents:

- `concave points_mean` or `concave_points_mean`
- `concave points_worst` or `concave_points_worst`

### Baseline versus selected model

Both rows use the same stratified 80/20 split, `random_state=42`, and the same preprocessing and LogisticRegression pipeline.

| Model | Features | Accuracy | Precision (M) | Recall (M) | F1 (M) | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 30 | 98.25% | 97.62% | 97.62% | 97.62% | `[[71, 1], [1, 41]]` |
| Selected | 17 | 97.37% | 95.35% | 97.62% | 96.47% | `[[70, 2], [1, 41]]` |

The 17-feature model gives up about 0.88 percentage points of accuracy on this single test split, but uses 13 fewer features and preserves malignant recall. The selected model is the final production model because it is simpler and matches the reduced feature schema used by the API.

These numbers are illustrative for one fixed split. They do not establish clinical reliability.

## 5. Final Model

The saved artifact is `models/model.joblib` and contains an `imblearn.pipeline.Pipeline`:

1. `OutlierClipper` — IQR clipping with boundaries learned from the training data.
2. `RobustScaler` — scaling fitted on the training data.
3. `SMOTE(random_state=42)` — applied only during training.
4. `LogisticRegression(max_iter=1000)` — final classifier.

Class mapping:

- `0` = Benign
- `1` = Malignant

At inference time, raw values for the 17 selected features are passed through the saved pipeline. SMOTE is not applied to inference data.

## 6. Project Structure

```text
task 1/
├── data/
│   └── data.csv
├── models/
│   └── model.joblib
├── notebooks/
│   └── 01_exploration.ipynb
├── production_app/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── logging_config.py
│   ├── main.py
│   ├── model_service.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── schemas.py
│   └── training.py
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── ui/
│   └── streamlit_app.py
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.example
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── run.py
├── uv.lock
└── README.md
```

## 7. Requirements and Setup

- Python 3.12, as specified in `.python-version`
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Docker Desktop / Docker Engine with Compose for the container workflow

Install dependencies:

```bash
uv sync
```

`uv sync` installs the development dependencies as well, including pytest, `httpx2`, Ruff, and Black.

For production-only dependencies:

```bash
uv sync --no-dev
```

### Environment variables

Copy `.env.example` to `.env` and adjust values if needed:

```bash
cp .env.example .env
```

```powershell
Copy-Item .env.example .env
```

Example:

```dotenv
DATA_PATH=data/data.csv
MODEL_PATH=models/model.joblib
METRICS_PATH=artifacts/metrics.json
API_URL=http://127.0.0.1:8000
LOG_LEVEL=INFO
```

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `DATA_PATH` | `data/data.csv` | training, UI stats | Dataset path |
| `MODEL_PATH` | `models/model.joblib` | training, API | Saved model artifact path |
| `METRICS_PATH` | `artifacts/metrics.json` | configuration | Reserved for future metrics export; current training prints metrics to the console/log |
| `API_URL` | `http://127.0.0.1:8000` | Streamlit UI | Base URL of the prediction API |
| `LOG_LEVEL` | `INFO` | all modules | Logging level |

Relative paths are resolved from the project root.

## 8. Run Guide

### 8.1 Train and recreate the model artifact

From the project root:

```bash
uv sync
uv run python -m production_app.main
```

Alternatively:

```bash
uv run python run.py
```

The command loads `data/data.csv`, trains the 17-feature pipeline, prints evaluation results, and writes `models/model.joblib`.

Running it again retrains the model and overwrites the artifact.

### 8.2 Start the API

```bash
uv run uvicorn production_app.api:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Health: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

For a non-reload production-style local run, omit `--reload`.

### 8.3 Start the Streamlit UI

With the API running:

```bash
uv run streamlit run ui/streamlit_app.py
```

The UI reads `API_URL` (default `http://127.0.0.1:8000`), sends the 17 features to `/predict`, and displays the predicted label and both class probabilities.

### 8.4 Docker

Start Docker Desktop / the Docker engine first. Compose uses two images:

- `Dockerfile.api` — FastAPI on port `8000`
- `Dockerfile.frontend` — Streamlit on port `8501`

If Compose reports `failed to connect to the docker API ... dockerDesktopLinuxEngine`, the Docker engine is not running. Start Docker Desktop and wait until `docker info` succeeds.

Both Dockerfiles use `python:3.12-slim` with Python 3.12.

Build and start:

```bash
docker compose build
docker compose up -d
```

Check status and logs:

```bash
docker compose ps
docker compose logs -f
```

Open:

- API health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs
- Streamlit UI: http://127.0.0.1:8501

Inside Compose, the UI reaches the API through the service name:

```yaml
environment:
  - API_URL=http://api:8000
```

Stop the stack:

```bash
docker compose down
```

The model is copied into the API image and the dataset into the UI image. Rebuild the affected image after retraining the model or changing the dataset:

```bash
docker compose build
docker compose up -d
```

### 8.5 Run the tests

```bash
uv run pytest
```

Run individual files:

```bash
uv run pytest tests/test_model.py
uv run pytest tests/test_api.py
```

Verbose output:

```bash
uv run pytest -v
```

Expected result: `9 passed`.

### 8.6 Code quality

```bash
uv run ruff check production_app tests
uv run black --check production_app tests
uv run pre-commit run --all-files
```

To apply fixes:

```bash
uv run ruff check production_app tests --fix
uv run black production_app tests
```

## 9. API Contract

### `GET /health`

Returns the model readiness state.

**Model available — `200 OK`**

```json
{
  "status": "healthy"
}
```

**Model unavailable — `503 Service Unavailable`**

```json
{
  "detail": {
    "status": "not_ready",
    "message": "Model is unavailable"
  }
}
```

The API still starts if the model artifact cannot be loaded. In that state, `/health` and `/predict` both return `503`.

### `POST /predict`

Request body: a JSON object with exactly the 17 selected features. All values must be numeric.

```json
{
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  "area_mean": 1001.0,
  "concavity_mean": 0.3001,
  "concave points_mean": 0.1471,
  "radius_se": 1.095,
  "area_se": 153.4,
  "compactness_se": 0.04904,
  "fractal_dimension_se": 0.006193,
  "radius_worst": 25.38,
  "texture_worst": 17.33,
  "perimeter_worst": 184.6,
  "area_worst": 2019.0,
  "smoothness_worst": 0.1622,
  "concavity_worst": 0.7119,
  "concave points_worst": 0.2654,
  "symmetry_worst": 0.4601
}
```

The two spaced names can also be sent as `concave_points_mean` and `concave_points_worst`.

**Success — `200 OK`**

```json
[1.3927386899315053e-07, 0.999999860726131]
```

- Index `0` is the benign probability.
- Index `1` is the malignant probability.
- The two values sum to `1.0`.
- The UI derives the display label from the higher probability.

**Validation error — `422 Unprocessable Entity`**

Returned when a selected feature is missing or a value is not numeric. Extra fields are ignored by the current Pydantic schema.

Example missing-field response (abridged):

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "texture_mean"],
      "msg": "Field required"
    }
  ]
}
```

**Model unavailable — `503 Service Unavailable`**

```json
{
  "detail": "Train the model first"
}
```

### How the UI connects

`ui/streamlit_app.py`:

1. Reads the 17 feature names from `production_app.config` and display ranges from `data/data.csv`.
2. Builds a JSON object from the sliders/number inputs.
3. Sends it to `{API_URL}/predict` with `Content-Type: application/json`.
4. Reads the two probabilities from the JSON list and derives the label (`Malignant` if the malignant probability is higher, otherwise `Benign`).

In Docker Compose, `API_URL` is set to `http://api:8000`. Outside Docker, the default is `http://127.0.0.1:8000`.

## 10. Tests

The test suite has 9 tests across two files.

### `tests/test_model.py`

- Confirms the selected feature count is 17.
- Loads the saved artifact and predicts with the selected feature shape without retraining.
- Exercises the shared prediction function.

### `tests/test_api.py`

- `GET /health` returns a healthy response when the model is available.
- `POST /predict` returns `200` with two class probabilities.
- Missing features return `422`.
- Malformed feature values return `422`.
- `/health` returns `503` when the model is unavailable.
- `/predict` returns `503` when the model is unavailable.

The model tests read the committed artifact and dataset; they do not retrain the model.

## 11. Reproducibility

The model artifact can be recreated from the committed dataset and source code.

1. Install the locked environment:

```bash
uv sync --frozen
```

2. Confirm the lockfile is current:

```bash
uv lock --check
```

3. Retrain and overwrite the artifact:

```bash
uv run python -m production_app.main
```

This writes `models/model.joblib`.

4. Optionally verify the API with the saved artifact:

```bash
uv run pytest
```

Reproducibility controls:

- `random_state=42` for the train/test split
- `random_state=42` for SMOTE
- 5-fold RFECV used to derive the fixed selected feature list
- Fixed selected feature list in `production_app/config.py`
- Locked dependency versions in `uv.lock`

The dataset is committed at `data/data.csv`. If it is missing, restore it before training. The model artifact is also committed, but the steps above recreate it if needed.

To rebuild the Docker images after retraining:

```bash
docker compose build
docker compose up -d
```

## 12. Limitations

- The reported metrics come from one stratified train/test split and are not a substitute for external validation.
- The project is educational and is not intended for clinical use.
- `METRICS_PATH` is currently reserved for future metrics export; the training workflow logs metrics but does not write a metrics JSON file.
