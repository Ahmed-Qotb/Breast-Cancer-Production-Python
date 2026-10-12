# Breast Cancer Classification — Machine Learning Pipeline

## 1. Project Overview

This project builds a reproducible machine learning pipeline to classify breast cancer tumors as **benign (B)** or **malignant (M)** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The project covers data exploration, preprocessing, model training, evaluation, and saving a trained model for later inference. The application is organized into reusable Python modules rather than relying exclusively on notebook code.

## 2. Dataset

The project uses the Wisconsin Diagnostic Breast Cancer dataset stored at `data/data.csv`.

- **Samples:** 569
- **Input features:** 30 numeric tumor measurements
- **Target:** `diagnosis`
- **Classes:** Benign (`B`) and malignant (`M`)

The dataset contains measurements derived from digitized images of fine needle aspirates of breast masses.

The `diagnosis` column is the target variable. It is separated from the input features before training.

## 3. Machine Learning Pipeline

The project uses the following workflow:

1. Load and validate the dataset.
2. Separate the input features and target.
3. Split the data into training and testing sets using a configurable test size and random seed, with stratification to preserve class proportions.
4. Calculate outlier-clipping boundaries from the training data and apply them to the training and testing sets.
5. Scale features using `RobustScaler`.
6. Apply `SMOTE` to address class imbalance during training.
7. Train a `LogisticRegression` classifier.
8. Evaluate the trained model using accuracy, a classification report, and a confusion matrix.
9. Save the fitted model to `models/model.joblib` for later use.

The scaler, SMOTE resampling step, and classifier are combined in an imbalanced-learn pipeline. SMOTE is applied during training, not to the held-out test data.

## 4. Project Structure

```text
task 1/
├── data/
│   └── data.csv
├── notebooks/
│   └── 01_exploration.ipynb
├── production_app/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   ├── model_service.py
│   └── main.py
├── models/
│   └── model.joblib
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── uv.lock
├── run.py
└── README.md
```

## 5. Requirements

- Python version specified in `.python-version`
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- Dependencies declared in `pyproject.toml` and locked in `uv.lock`

## 6. Installation and Setup

Open a terminal in the project root.

### Install dependencies

```bash
uv sync
```

### Configure environment variables

Copy `.env.example` to `.env` and adjust the settings if necessary.

Example configuration:

```dotenv
ENVIRONMENT=development
LOG_LEVEL=INFO
DATA_PATH=root\data\data.csv
```

The dataset path defaults to `data/data.csv`. If `DATA_PATH` is configured, it must point to the dataset file.

## 7. Run the Application

### Train and evaluate the model

```bash
uv run python -m production_app.main
```

Alternatively, use the root entry point:

```bash
uv run python run.py
```

The application loads the dataset, splits the data, trains the model, prints evaluation results, and saves the fitted model to:

```text
models/model.joblib
```

Running the training workflow again retrains the model and updates the saved artifact.

## 8. Evaluation

The model is evaluated on the held-out test set using:

- **Accuracy** — the proportion of correct predictions.
- **Classification report** — precision, recall, and F1-score for each class.
- **Confusion matrix** — the distribution of correct and incorrect predictions.

### Example results

One successful run produced the following results:

| Metric | Result |
|---|---:|
| Test samples | 114 |
| Accuracy | 98.25% |
| Correct predictions | 112 |
| Incorrect predictions | 2 |

Confusion matrix, with rows representing actual classes and columns representing predicted classes in the order `B`, `M`:

```text
[[71, 1],
 [ 1, 41]]
```

These results describe one train/test split with a fixed random seed. They do not establish clinical reliability or guarantee the same performance in other settings.

## 9. Code Quality

The project uses Ruff and Black for linting and formatting, and pre-commit to run quality checks.

Run Ruff:

```bash
uv run ruff check production_app
```

Run Black's formatting check:

```bash
uv run black --check production_app
```

Run all configured pre-commit hooks:

```bash
uv run pre-commit run --all-files
```

Format the project when needed:

```bash
uv run ruff check production_app --fix
uv run black production_app
```
