import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow the app to import production_app when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production_app.config import FEATURES, config

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
)


@st.cache_data
def get_feature_stats() -> pd.DataFrame:
    data = pd.read_csv(config.data_path)

    return data[FEATURES].agg(["min", "max", "mean"]).T


def sync_from_slider(feature: str) -> None:
    st.session_state[f"{feature}_number"] = st.session_state[f"{feature}_slider"]


def sync_from_number(feature: str) -> None:
    st.session_state[f"{feature}_slider"] = st.session_state[f"{feature}_number"]


def get_step_and_format(min_value: float, max_value: float) -> tuple[float, str]:
    span = max_value - min_value

    if span >= 100:
        return 1.0, "%.1f"
    if span >= 10:
        return 0.1, "%.2f"
    if span >= 1:
        return 0.01, "%.2f"

    return 0.001, "%.4f"


st.title("Breast Cancer Prediction")
st.write("Drag a slider or type a value, then press Predict.")

try:
    stats = get_feature_stats()
except OSError:
    stats = pd.DataFrame(
        {"min": 0.0, "max": 1.0, "mean": 0.5},
        index=FEATURES,
    )

for feature in FEATURES:
    slider_key = f"{feature}_slider"
    number_key = f"{feature}_number"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = float(stats.loc[feature, "mean"])

    if number_key not in st.session_state:
        st.session_state[number_key] = float(stats.loc[feature, "mean"])

feature_columns = st.columns(3)

for index, feature in enumerate(FEATURES):
    min_value = float(stats.loc[feature, "min"])
    max_value = float(stats.loc[feature, "max"])
    step, number_format = get_step_and_format(min_value, max_value)
    label = feature.replace("_", " ").title()

    with feature_columns[index % 3]:
        st.caption(label)

        slider_column, number_column = st.columns([3, 1])

        with slider_column:
            st.slider(
                label,
                min_value=min_value,
                max_value=max_value,
                step=step,
                format=number_format,
                key=f"{feature}_slider",
                label_visibility="collapsed",
                on_change=sync_from_slider,
                args=(feature,),
            )

        with number_column:
            st.number_input(
                label,
                min_value=min_value,
                max_value=max_value,
                step=step,
                format=number_format,
                key=f"{feature}_number",
                label_visibility="collapsed",
                on_change=sync_from_number,
                args=(feature,),
            )

if st.button("Predict", type="primary", use_container_width=True):
    values = {
        feature: float(st.session_state[f"{feature}_number"]) for feature in FEATURES
    }
    payload = json.dumps(values).encode("utf-8")

    request = urllib.request.Request(
        f"{config.api_url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with (
            st.spinner("Calling the prediction API..."),
            urllib.request.urlopen(request) as response,
        ):
            probabilities = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        st.error(f"API returned {error.code}: {error.read().decode('utf-8')}")
    except urllib.error.URLError as error:
        st.error(f"Could not reach the API at {config.api_url}/predict")
        st.error("Start it with: uv run uvicorn production_app.api:app --reload")
        st.error(str(error))
    else:
        benign, malignant = probabilities
        predicted_label = "Malignant" if malignant > benign else "Benign"

        benign_active = "active" if predicted_label == "Benign" else ""
        malignant_active = "active" if predicted_label == "Malignant" else ""

        benign_badge = (
            '<span class="result-badge">Predicted</span>'
            if predicted_label == "Benign"
            else ""
        )
        malignant_badge = (
            '<span class="result-badge">Predicted</span>'
            if predicted_label == "Malignant"
            else ""
        )

        st.divider()
        st.subheader("Prediction result")

        st.html(
            f"""
            <style>
            .result-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 0.75rem;
            }}
            .result-card {{
                position: relative;
                padding: 1.2rem 1.35rem;
                border-radius: 16px;
                border: 1px solid rgba(15, 23, 42, 0.08);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }}
            .result-card.prediction {{
                background: linear-gradient(135deg, #eef2ff, #e0e7ff);
                color: #312e81;
                border-color: #c7d2fe;
            }}
            .result-card.benign {{
                background: linear-gradient(135deg, #ecfdf5, #d1fae5);
                color: #065f46;
                border-color: #a7f3d0;
            }}
            .result-card.malignant {{
                background: linear-gradient(135deg, #fef2f2, #fee2e2);
                color: #991b1b;
                border-color: #fecaca;
            }}
            .result-card.active {{
                transform: translateY(-3px);
                border-width: 3px;
            }}
            .result-card.benign.active {{
                border-color: #10b981;
                box-shadow: 0 12px 24px rgba(16, 185, 129, 0.25);
            }}
            .result-card.malignant.active {{
                border-color: #ef4444;
                box-shadow: 0 12px 24px rgba(239, 68, 68, 0.25);
            }}
            .result-label {{
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                opacity: 0.8;
            }}
            .result-value {{
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.15;
                margin-top: 0.4rem;
            }}
            .result-note {{
                font-size: 0.82rem;
                margin-top: 0.35rem;
                opacity: 0.75;
            }}
            .result-badge {{
                position: absolute;
                top: 0.8rem;
                right: 0.8rem;
                font-size: 0.68rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                padding: 0.2rem 0.5rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.75);
            }}
            @media (max-width: 900px) {{
                .result-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
            </style>

            <div class="result-grid">
                <div class="result-card prediction">
                    <div class="result-label">Prediction</div>
                    <div class="result-value">{predicted_label}</div>
                    <div class="result-note">Highest probability</div>
                </div>

                <div class="result-card benign {benign_active}">
                    {benign_badge}
                    <div class="result-label">Benign probability</div>
                    <div class="result-value">{benign:.2%}</div>
                    <div class="result-note">Class 0 · B</div>
                </div>

                <div class="result-card malignant {malignant_active}">
                    {malignant_badge}
                    <div class="result-label">Malignant probability</div>
                    <div class="result-value">{malignant:.2%}</div>
                    <div class="result-note">Class 1 · M</div>
                </div>
            </div>
            """
        )
