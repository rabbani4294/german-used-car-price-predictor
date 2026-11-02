from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_price_pipeline.joblib"
DATASET_PATH = BASE_DIR / "cleaned_cars.csv"

FEATURE_COLUMNS = [
    "mileage", "hp", "year", "make", "model", "fuel", "gear", "offerType"
]

COLORS = {
    "background": "#0B1220",
    "surface": "#151F30",
    "blue": "#2563EB",
    "cyan": "#06B6D4",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "green": "#22C55E",
}


st.set_page_config(
    page_title="German Car Price Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root { color-scheme: dark; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp {
            background:
              radial-gradient(circle at 82% 3%, rgba(37,99,235,.16), transparent 27rem),
              radial-gradient(circle at 8% 25%, rgba(6,182,212,.08), transparent 25rem),
              #0B1220;
            color: #F8FAFC;
        }
        [data-testid="stHeader"] { background: rgba(11,18,32,.88); }
        [data-testid="stToolbar"] { right: 1rem; }
        .block-container { max-width: 1440px; padding-top: 1.7rem; padding-bottom: 3rem; }
        h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; }
        .hero {
            position: relative; overflow: hidden; min-height: 345px; padding: 3.1rem 3rem;
            border: 1px solid rgba(148,163,184,.2); border-radius: 26px;
            background:
              linear-gradient(90deg, rgba(8,15,28,.98) 0%, rgba(8,15,28,.88) 43%, rgba(8,15,28,.28) 100%),
              linear-gradient(0deg, rgba(8,15,28,.25), rgba(8,15,28,.25)),
              url('https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=1800&q=85') center 58%/cover;
            box-shadow: 0 24px 70px rgba(0,0,0,.28);
            display: flex; align-items: center;
        }
        .hero-content { max-width: 690px; position: relative; z-index: 1; }
        .eyebrow { color: #67E8F9; font-size: .76rem; font-weight: 700; letter-spacing: .22em; text-transform: uppercase; }
        .hero h1 { color: #F8FAFC; font-size: clamp(2.1rem, 4.4vw, 4.1rem); line-height: 1.04; margin: .75rem 0 1rem; }
        .hero p { color: #CBD5E1; font-size: 1.04rem; line-height: 1.75; max-width: 590px; }
        .hero-tag { display: inline-block; margin-top: 1rem; padding: .55rem .9rem; border-radius: 999px;
            border: 1px solid rgba(103,232,249,.32); background: rgba(6,182,212,.1); color: #A5F3FC; font-size: .82rem; }
        .section-kicker { color: #67E8F9; font-size: .72rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; margin-bottom: .35rem; }
        .section-title { font-family: 'Manrope',sans-serif; font-size: 1.65rem; font-weight: 800; color: #F8FAFC; margin: 0 0 .35rem; }
        .section-copy { color: #94A3B8; margin-bottom: 1.15rem; }
        .metric-card {
            background: linear-gradient(145deg, rgba(21,31,48,.96), rgba(15,25,41,.94));
            border: 1px solid rgba(148,163,184,.16); border-radius: 18px; padding: 1.15rem 1.25rem;
            min-height: 112px; box-shadow: 0 12px 30px rgba(0,0,0,.14);
        }
        .metric-label { color: #94A3B8; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; }
        .metric-value { color: #F8FAFC; font-family: 'Manrope',sans-serif; font-size: 1.75rem; font-weight: 800; margin-top: .45rem; }
        .metric-sub { color: #64748B; font-size: .75rem; margin-top: .15rem; }
        .panel {
            background: linear-gradient(145deg, rgba(21,31,48,.92), rgba(13,23,38,.96));
            border: 1px solid rgba(148,163,184,.16); border-radius: 20px; padding: 1.35rem 1.45rem;
            box-shadow: 0 14px 36px rgba(0,0,0,.16);
        }
        .result-price { font-family: 'Manrope',sans-serif; font-size: clamp(2.3rem, 5vw, 3.8rem); font-weight: 800;
            letter-spacing: -.055em; color: #F8FAFC; margin: .2rem 0; }
        .result-label { color: #67E8F9; text-transform: uppercase; letter-spacing: .15em; font-size: .72rem; font-weight: 700; }
        .spec-label { color: #94A3B8; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }
        .spec-value { color: #F8FAFC; font-size: .98rem; font-weight: 600; margin-top: .2rem; overflow-wrap: anywhere; }
        .spec-box { background: rgba(30,41,59,.55); border: 1px solid rgba(148,163,184,.12); border-radius: 13px; padding: .85rem; min-height: 72px; }
        .note { color: #94A3B8; font-size: .82rem; line-height: 1.6; }
        .footer { border-top: 1px solid rgba(148,163,184,.15); margin-top: 3rem; padding-top: 1.2rem; color: #64748B; font-size: .78rem; text-align: center; }
        div.stButton > button[kind="primary"] {
            width: 100%; min-height: 3.2rem; border: 0; border-radius: 12px;
            color: white; font-weight: 800; letter-spacing: .08em;
            background: linear-gradient(100deg, #2563EB, #0891B2);
            box-shadow: 0 10px 28px rgba(37,99,235,.28); transition: transform .2s ease, box-shadow .2s ease;
        }
        div.stButton > button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 14px 34px rgba(6,182,212,.25); }
        div.stButton > button[kind="secondary"] { border-radius: 11px; border: 1px solid rgba(148,163,184,.25); }
        [data-testid="stWidgetLabel"] p { color: #CBD5E1; font-weight: 600; }
        [data-baseweb="select"] > div, [data-testid="stNumberInput"] input {
            background: #111D2E; border-color: rgba(148,163,184,.24); border-radius: 10px; color: #F8FAFC;
        }
        [data-testid="stExpander"] { background: rgba(21,31,48,.65); border: 1px solid rgba(148,163,184,.15); border-radius: 14px; }
        @media(max-width: 700px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .hero { min-height: 390px; padding: 2rem 1.35rem; background-position: 62% center; }
            .hero p { font-size: .92rem; }
            .metric-value { font-size: 1.45rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. Place it in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH.name}. It is required for real dataset statistics and dependent dropdowns."
        )
    data = pd.read_csv(DATASET_PATH)
    required = set(FEATURE_COLUMNS + ["price"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    return data


@st.cache_data
def get_training_reference_data(data: pd.DataFrame) -> tuple[list[str], dict[str, list[str]], float, float, float]:
    # Recreate the notebook's 80/20 split so dropdown categories and reference prices
    # are based on the same training partition used by the saved model.
    X = data[FEATURE_COLUMNS].copy()
    y = data["price"].copy()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    makes = sorted(X_train["make"].dropna().astype(str).unique().tolist())
    make_model_map: dict[str, list[str]] = {}
    for make in makes:
        models = (
            X_train.loc[X_train["make"].astype(str) == make, "model"]
            .fillna("Unknown")
            .astype(str)
            .unique()
            .tolist()
        )
        make_model_map[make] = sorted(models)

    return (
        makes,
        make_model_map,
        float(y_train.mean()),
        float(y_train.median()),
        float(y_train.quantile(0.95)),
    )


def reset_model_selection() -> None:
    st.session_state["model_select"] = None


def reset_form() -> None:
    for key in [
        "make_select", "model_select", "year_input", "mileage_input",
        "hp_input", "fuel_select", "gear_select", "offer_select",
        "last_prediction",
    ]:
        st.session_state.pop(key, None)


def make_gauge(prediction: float, upper_bound: float) -> go.Figure:
    maximum = max(upper_bound, prediction * 1.12, 1000.0)
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"prefix": "€", "valueformat": ",.0f", "font": {"size": 38, "color": COLORS["text"]}},
            title={"text": "ESTIMATED VEHICLE VALUE", "font": {"size": 13, "color": COLORS["muted"]}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, maximum], "tickprefix": "€", "tickformat": ",.0f",
                         "tickcolor": COLORS["muted"], "tickfont": {"color": COLORS["muted"], "size": 10}},
                "bar": {"color": COLORS["cyan"], "thickness": 0.28},
                "bgcolor": COLORS["surface"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, maximum * 0.55], "color": "rgba(37,99,235,.16)"},
                    {"range": [maximum * 0.55, maximum], "color": "rgba(6,182,212,.12)"},
                ],
                "threshold": {"line": {"color": "#F8FAFC", "width": 3}, "thickness": 0.8, "value": prediction},
            },
        )
    )
    figure.update_layout(
        height=330, margin=dict(l=25, r=25, t=65, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": COLORS["text"]},
    )
    return figure


def make_comparison_chart(prediction: float, mean_price: float, median_price: float) -> go.Figure:
    labels = ["Predicted vehicle", "Dataset mean", "Dataset median"]
    values = [prediction, mean_price, median_price]
    colors = [COLORS["cyan"], COLORS["blue"], "#64748B"]
    figure = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"€{value:,.2f}" for value in values],
            textposition="outside",
            textfont={"color": COLORS["text"], "size": 12},
            cliponaxis=False,
        )
    )
    figure.update_layout(
        height=260, margin=dict(l=10, r=90, t=15, b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": True, "gridcolor": "rgba(148,163,184,.12)", "zeroline": False,
               "tickprefix": "€", "tickformat": ",.0f", "tickfont": {"color": COLORS["muted"]}},
        yaxis={"showgrid": False, "tickfont": {"color": COLORS["text"], "size": 12}},
        font={"family": "DM Sans, sans-serif", "color": COLORS["text"]},
        showlegend=False,
    )
    return figure


def render_metric(label: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-sub">{subtext}</div></div>',
        unsafe_allow_html=True,
    )


def render_spec(label: str, value: str) -> None:
    st.markdown(
        f'<div class="spec-box"><div class="spec-label">{label}</div>'
        f'<div class="spec-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()

    try:
        model = load_model()
        dataset = load_dataset()
        makes, make_model_map, mean_price, median_price, p95_price = get_training_reference_data(dataset)
    except Exception as error:
        st.error(f"Application setup error: {error}")
        st.info(
            "Check that app.py, car_price_pipeline.joblib, and cleaned_cars.csv are in the same project folder. "
            "Install the pinned dependencies from requirements.txt."
        )
        st.stop()

    st.markdown(
        """
        <section class="hero">
          <div class="hero-content">
            <div class="eyebrow">Automotive intelligence / Machine learning</div>
            <h1>GERMAN CAR<br>PRICE PREDICTOR</h1>
            <p>Intelligent Machine Learning for Used Car Valuation. Explore a data-driven estimate for your next vehicle using a tuned Ridge Regression pipeline.</p>
            <span class="hero-tag">PRECISION-DRIVEN VEHICLE VALUATION</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="section-kicker">The model at a glance</div><div class="section-title">Performance overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Evaluation results from the held-out test set.</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metrics = [
        ("R² SCORE", "88.04%", "Explained variance"),
        ("RMSE", "€6,530.73", "Root mean squared error"),
        ("MAE", "€2,846.87", "Mean absolute error"),
        ("CLEANED RECORDS", f"{len(dataset):,}", "Rows after documented cleaning"),
    ]
    for column, metric in zip(metric_cols, metrics):
        with column:
            render_metric(*metric)
    st.caption("These are test-set evaluation metrics. They describe overall model performance and do not guarantee the accuracy of an individual vehicle estimate.")

    st.write("")
    st.markdown('<div class="section-kicker">Configure your vehicle</div><div class="section-title">Vehicle specification</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Enter the details below to generate an estimated market price.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**01  Vehicle identity**")
        identity_cols = st.columns(3)
        with identity_cols[0]:
            selected_make = st.selectbox(
                "Manufacturer", options=makes, key="make_select",
                on_change=reset_model_selection,
            )
        available_models = make_model_map.get(selected_make, [])
        with identity_cols[1]:
            model_options: list[str | None] = [None] + available_models
            selected_model = st.selectbox(
                "Model", options=model_options, key="model_select",
                format_func=lambda value: "Select a model" if value is None else value,
            )
        with identity_cols[2]:
            year = st.number_input("Manufacturing year", min_value=int(dataset["year"].min()), max_value=int(dataset["year"].max()), value=2018, step=1, key="year_input")

        st.markdown("**02  Technical specifications**")
        technical_cols = st.columns(4)
        with technical_cols[0]:
            mileage = st.number_input("Mileage (KM)", min_value=0, max_value=2_000_000, value=50_000, step=1_000, key="mileage_input")
        with technical_cols[1]:
            hp = st.number_input("Horsepower (HP)", min_value=1, max_value=2_000, value=120, step=5, key="hp_input")
        with technical_cols[2]:
            fuel_options = sorted(dataset["fuel"].dropna().astype(str).unique().tolist())
            fuel = st.selectbox("Fuel type", fuel_options, key="fuel_select")
        with technical_cols[3]:
            gear_options = sorted(dataset["gear"].dropna().astype(str).unique().tolist())
            gear = st.selectbox("Transmission", gear_options, key="gear_select")

        st.markdown("**03  Selling information**")
        offer_options = sorted(dataset["offerType"].dropna().astype(str).unique().tolist())
        offer_type = st.selectbox("Offer type", offer_options, key="offer_select")

        action_cols = st.columns([3, 1])
        with action_cols[0]:
            predict_clicked = st.button("PREDICT CAR PRICE", type="primary", use_container_width=True)
        with action_cols[1]:
            st.button("RESET FORM", use_container_width=True, on_click=reset_form)

    if predict_clicked:
        if selected_model is None:
            st.warning("Please select a vehicle model before generating a prediction.")
        elif not np.isfinite(float(mileage)) or not np.isfinite(float(hp)):
            st.error("Please enter valid numeric vehicle specifications.")
        else:
            input_data = pd.DataFrame([{
                "mileage": int(mileage),
                "hp": int(hp),
                "year": int(year),
                "make": str(selected_make),
                "model": str(selected_model),
                "fuel": str(fuel),
                "gear": str(gear),
                "offerType": str(offer_type),
            }], columns=FEATURE_COLUMNS)

            try:
                with st.spinner("Analyzing vehicle specifications..."):
                    prediction = float(model.predict(input_data)[0])
                if not np.isfinite(prediction):
                    raise ValueError("The model returned a non-finite prediction.")
                prediction = max(0.0, prediction)
                st.session_state["last_prediction"] = {
                    "price": prediction,
                    "vehicle": {
                        "Manufacturer": str(selected_make),
                        "Model": str(selected_model),
                        "Year": int(year),
                        "Mileage": f"{int(mileage):,} KM",
                        "Horsepower": f"{int(hp):,} HP",
                        "Fuel": str(fuel),
                        "Transmission": str(gear),
                        "Offer type": str(offer_type),
                    },
                    "timestamp": datetime.now().astimezone().strftime("%d %b %Y, %H:%M %Z"),
                }
            except Exception as error:
                st.error(f"Prediction could not be generated: {error}")

    result = st.session_state.get("last_prediction")
    if result:
        st.write("")
        st.markdown('<div class="section-kicker">Your valuation report</div><div class="section-title">Prediction result</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-copy">A model-based estimate using the saved preprocessing and regression pipeline.</div>', unsafe_allow_html=True)

        left, right = st.columns([1.05, 1.25], gap="large")
        with left:
            st.markdown(
                f'<div class="panel"><div class="result-label">Estimated vehicle price</div>'
                f'<div class="result-price">€{result["price"]:,.2f}</div>'
                f'<div class="note">Generated {result["timestamp"]}</div></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_gauge(result["price"], p95_price), use_container_width=True, config={"displayModeBar": False})
        with right:
            st.markdown('<div class="panel"><div class="section-kicker">Market context</div><div class="section-title">Price comparison</div>', unsafe_allow_html=True)
            st.plotly_chart(make_comparison_chart(result["price"], mean_price, median_price), use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                '<div class="note">Dataset mean and median are calculated from the training partition recreated with the original 80/20 split and random_state=42. They are broad dataset references, not direct valuations of this specific vehicle.</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("**Vehicle profile**")
        specs = list(result["vehicle"].items())
        spec_cols = st.columns(4)
        for index, (label, value) in enumerate(specs):
            with spec_cols[index % 4]:
                render_spec(label, str(value))

        st.caption("This estimate is not an official appraisal, dealer quote, or guarantee of a sale price. Actual market value may vary with condition, service history, equipment, location, and market demand.")

    with st.expander("About this project"):
        st.write(
            "This application uses a pre-trained Tuned Ridge Regression pipeline. "
            "The saved pipeline performs its own missing-value imputation, numerical scaling, and categorical encoding. "
            "The application does not retrain the model or preprocess prediction inputs separately."
        )
        st.write(
            f"Dataset: {len(dataset):,} cleaned records. Target: price. "
            "The data cleaning follows the accompanying project notebook: duplicate removal, "
            "suspicious mileage and horsepower values marked missing, and exclusion of the specified non-passenger makes."
        )

    with st.expander("Model performance details"):
        detail_cols = st.columns(3)
        with detail_cols[0]:
            render_metric("R²", "0.8804", "Test set")
        with detail_cols[1]:
            render_metric("RMSE", "€6,530.73", "Test set")
        with detail_cols[2]:
            render_metric("MAE", "€2,846.87", "Test set")
        st.markdown(
            '<div class="note">R² measures the proportion of target variance explained by the model. RMSE penalizes larger errors more strongly, while MAE represents the average absolute prediction error. None of these metrics guarantees an individual prediction.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer">GERMAN CAR PRICE PREDICTOR &nbsp;|&nbsp; MACHINE LEARNING VALUATION PLATFORM<br>'
        'Built with Python, Streamlit, Pandas, Plotly and a tuned Ridge Regression pipeline.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
