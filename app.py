import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="VoltIQ | Demand Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "electricity_demand.csv"
MODEL_FILE = "best_spike_model.joblib"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature_c", "humidity_pct", "demand_lag_1_mw",
    "demand_rolling_3h_mw", "demand_rolling_24h_mw"
]
TARGET = "spike_next_hour"

# ----------------------------- LUXURY PURPLE UI -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(126, 77, 255, .20), transparent 34%),
    radial-gradient(ellipse at 95% 12%, rgba(194, 104, 255, .12), transparent 28%),
    linear-gradient(135deg, #0b0816 0%, #100c20 48%, #0b0915 100%);
  color: #f5f0ff;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #17102b 0%, #100c1e 100%);
  border-right: 1px solid rgba(190,160,255,.15);
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.5px; }
.hero {
  padding: 30px 34px; border-radius: 24px; margin-bottom: 22px;
  background: linear-gradient(115deg, rgba(103,64,190,.30), rgba(35,24,61,.75));
  border: 1px solid rgba(190,160,255,.22);
  box-shadow: 0 18px 60px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.06);
}
.eyebrow { color:#c6a8ff; text-transform:uppercase; letter-spacing:2.2px; font-size:11px; font-weight:700; }
.hero-title { font-family:'Space Grotesk'; font-size:36px; font-weight:700; margin:7px 0 4px; color:#fff; }
.hero-sub { color:#c5badb; font-size:14px; }
.metric-card {
  background: linear-gradient(145deg, rgba(38,27,64,.94), rgba(22,17,39,.94));
  border:1px solid rgba(190,160,255,.16); border-radius:18px;
  padding:19px 20px; min-height:112px; box-shadow:0 12px 35px rgba(0,0,0,.16);
}
.metric-label { color:#aa9cbe; font-size:12px; text-transform:uppercase; letter-spacing:1.2px; }
.metric-value { color:#fff; font-family:'Space Grotesk'; font-size:27px; font-weight:700; margin-top:7px; }
.metric-note { color:#b79ae9; font-size:11px; margin-top:3px; }
.section-card {
  background:rgba(25,19,42,.78); border:1px solid rgba(190,160,255,.13);
  border-radius:20px; padding:20px 22px; margin:10px 0 18px;
}
div[data-testid="stButton"] > button, div[data-testid="stFormSubmitButton"] > button {
  background:linear-gradient(110deg,#8957ef,#bd79ff); color:white; border:0;
  border-radius:12px; font-weight:700; min-height:44px;
  box-shadow:0 8px 22px rgba(137,87,239,.25); transition:all .2s ease;
}
div[data-testid="stButton"] > button:hover { transform:translateY(-2px); box-shadow:0 12px 28px rgba(137,87,239,.38); }
.stTabs [data-baseweb="tab"] { color:#b9a9d1; }
.stTabs [aria-selected="true"] { color:#d8baff !important; }
div[data-testid="stDataFrame"] { border:1px solid rgba(190,160,255,.15); border-radius:14px; overflow:hidden; }
div[data-testid="stAlert"] { border-radius:14px; }
small, .muted { color:#a99bbd; }
hr { border-color:rgba(190,160,255,.13); }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- HELPERS -----------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(int)
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1.")
    return data

def build_models():
    return {
        "Baseline (Most Frequent)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=260, max_depth=9, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=42
            ))
        ])
    }

def metric_row(name, y_true, pred, proba=None):
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1 Score": f1_score(y_true, pred, zero_division=0),
    }
    if proba is not None and len(np.unique(y_true)) == 2:
        row["ROC AUC"] = roc_auc_score(y_true, proba)
    else:
        row["ROC AUC"] = np.nan
    return row

def make_input_frame(hour, day, month, temp, humidity, demand, roll3, roll24):
    return pd.DataFrame([{
        "hour": hour, "day_of_week": day, "month": month,
        "is_weekend": int(day >= 5), "temperature_c": temp,
        "humidity_pct": humidity, "demand_lag_1_mw": demand,
        "demand_rolling_3h_mw": roll3, "demand_rolling_24h_mw": roll24
    }], columns=FEATURES)

# ----------------------------- HEADER -----------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">TEAM 11 &nbsp; • &nbsp; MACHINE LEARNING / ENERGY ANALYTICS</div>
  <div class="hero-title">⚡ VoltIQ <span style="color:#c18cff">Demand Intelligence</span></div>
  <div class="hero-sub">Electricity Demand Spike Alert Using ML · Predict the next-hour spike. Prepare before demand surges.</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------- DATA + TRAINING -----------------------------
if not os.path.exists(DATA_FILE):
    st.error(f"Dataset not found: {DATA_FILE}. Keep the CSV in the same folder as app.py.")
    st.stop()

try:
    data = load_data(DATA_FILE)
except Exception as exc:
    st.error(f"Dataset validation failed: {exc}")
    st.stop()

# Chronological holdout: earliest 80% train, latest 20% test.
split_idx = int(len(data) * 0.80)
train = data.iloc[:split_idx].copy()
test = data.iloc[split_idx:].copy()
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

if y_train.nunique() < 2:
    st.error("Training data must contain both target classes (0 and 1). Please use a larger or more varied dataset.")
    st.stop()

@st.cache_resource
def train_and_evaluate(train_x, train_y, test_x, test_y):
    models = build_models()
    fitted, rows, predictions = {}, [], {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        proba = model.predict_proba(test_x)[:, 1] if hasattr(model, "predict_proba") else None
        fitted[name] = model
        predictions[name] = pred
        rows.append(metric_row(name, test_y, pred, proba))
    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    return fitted, results, predictions

with st.spinner("Training and evaluating models on the chronological holdout set..."):
    fitted_models, results, all_predictions = train_and_evaluate(
        X_train, y_train, X_test, y_test
    )

# Select strongest non-baseline model by F1.
eligible = results[results["Model"] != "Baseline (Most Frequent)"]
best_name = eligible.iloc[0]["Model"] if not eligible.empty else results.iloc[0]["Model"]
best_model = fitted_models[best_name]
best_predictions = all_predictions[best_name]

# Save reproducible model artifact.
joblib.dump({"model": best_model, "features": FEATURES, "target": TARGET, "model_name": best_name}, MODEL_FILE)

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown("## ◈ VOLTIQ")
    st.caption("ENERGY INTELLIGENCE SUITE")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Overview", "Live Spike Predictor", "Model Performance", "Dataset Explorer"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**PROJECT DETAILS**")
    st.caption("Team 11")
    st.caption("Electricity Demand Spike Alert Using ML")
    st.caption("Route A · Supervised Learning")
    st.markdown("---")
    st.caption("Demo dataset: synthetic · Replace with mentor-approved data before final submission.")

# ----------------------------- KPI ROW -----------------------------
spike_rate = float(data[TARGET].mean() * 100)
latest_demand = float(data["demand_lag_1_mw"].iloc[-1])
avg_demand = float(data["demand_lag_1_mw"].mean())
f1_best = float(results.loc[results["Model"] == best_name, "F1 Score"].iloc[0])

k1, k2, k3, k4 = st.columns(4)
for col, label, value, note in [
    (k1, "TOTAL RECORDS", f"{len(data):,}", "Hourly observations"),
    (k2, "SPIKE RATE", f"{spike_rate:.1f}%", "Positive target share"),
    (k3, "AVG. DEMAND", f"{avg_demand:.1f} MW", "Across demo dataset"),
    (k4, "BEST MODEL F1", f"{f1_best:.3f}", best_name)
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

st.write("")

# ----------------------------- PAGES -----------------------------
if page == "Overview":
    st.subheader("Demand overview")
    st.caption("Explore the hourly demand pattern and the model's next-hour alert history.")
    c1, c2 = st.columns([1.65, 1])
    with c1:
        plot_df = data.tail(168).copy()
        fig = px.line(plot_df, x="timestamp", y="demand_lag_1_mw", title="Hourly electricity demand · Last 7 days",
                      labels={"timestamp":"Time", "demand_lag_1_mw":"Demand (MW)"})
        fig.update_traces(line=dict(color="#b58aff", width=2.5), fill="tozeroy", fillcolor="rgba(160,110,255,.08)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10,r=10,t=55,b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        counts = data[TARGET].value_counts().rename(index={0:"Normal",1:"Spike alert"}).reset_index()
        counts.columns = ["Class","Records"]
        fig2 = px.pie(counts, names="Class", values="Records", hole=.68,
                      color="Class", color_discrete_map={"Normal":"#8b72c7","Spike alert":"#e2a0ff"},
                      title="Target distribution")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10,r=10,t=55,b=10), height=360, legend=dict(orientation="h",y=-.08))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Latest model evaluation")
    st.dataframe(results.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1 Score","ROC AUC"]}), use_container_width=True, hide_index=True)
    st.info("The model estimates whether demand in the next hour will exceed the recent 24-hour average by the demo threshold. This is an educational alert, not an operational grid-control system.")

elif page == "Live Spike Predictor":
    st.subheader("Live spike predictor")
    st.caption("Enter current conditions to estimate the probability of a demand spike in the next hour.")
    with st.form("prediction_form"):
        a,b,c = st.columns(3)
        with a:
            hour_in = st.slider("Hour of day", 0, 23, 18)
            day_in = st.selectbox("Day of week", options=list(range(7)), format_func=lambda x:["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x], index=4)
            month_in = st.selectbox("Month", list(range(1,13)), index=5)
        with b:
            temp_in = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=32.0, step=0.5)
            humidity_in = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
            demand_in = st.number_input("Current demand (MW)", min_value=0.0, max_value=5000.0, value=float(round(latest_demand,1)), step=5.0)
        with c:
            roll3_in = st.number_input("Recent 3-hour average (MW)", min_value=0.0, max_value=5000.0, value=float(round(data["demand_rolling_3h_mw"].iloc[-1],1)), step=5.0)
            roll24_in = st.number_input("Recent 24-hour average (MW)", min_value=0.0, max_value=5000.0, value=float(round(data["demand_rolling_24h_mw"].iloc[-1],1)), step=5.0)
            st.caption("Use consistent MW units for all demand inputs.")
        submitted = st.form_submit_button("⚡ ANALYZE NEXT-HOUR RISK", use_container_width=True)

    if submitted:
        sample = make_input_frame(hour_in, day_in, month_in, temp_in, humidity_in, demand_in, roll3_in, roll24_in)
        probability = float(best_model.predict_proba(sample)[0,1]) if hasattr(best_model,"predict_proba") else float(best_model.predict(sample)[0])
        prediction = int(probability >= 0.5)
        if prediction:
            st.error(f"⚠️ SPIKE ALERT · Estimated probability: {probability:.1%}")
            st.write("The model flags elevated next-hour demand risk. Review staffing and capacity plans using approved operational procedures.")
        else:
            st.success(f"✓ NORMAL DEMAND · Estimated spike probability: {probability:.1%}")
            st.write("The model does not flag a spike for the supplied conditions.")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=probability*100, number={"suffix":"%","font":{"color":"white"}},
            title={"text":"Next-hour spike probability","font":{"color":"#d8c5ff"}},
            gauge={"axis":{"range":[0,100],"tickcolor":"#cbb9e9"},
                   "bar":{"color":"#c18cff"},
                   "bgcolor":"rgba(255,255,255,.04)",
                   "steps":[{"range":[0,50],"color":"rgba(105,75,160,.18)"},{"range":[50,100],"color":"rgba(220,130,255,.18)"}],
                   "threshold":{"line":{"color":"#f0c5ff","width":4},"thickness":.75,"value":50}}
        ))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=290, margin=dict(l=20,r=20,t=60,b=10))
        st.plotly_chart(gauge, use_container_width=True)
        with st.expander("View validated input sent to the model"):
            st.dataframe(sample, use_container_width=True, hide_index=True)

elif page == "Model Performance":
    st.subheader("Model performance lab")
    st.caption("All models use the same chronological 80/20 train-test split.")
    st.markdown("### Model comparison")
    st.dataframe(results.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1 Score","ROC AUC"]}), use_container_width=True, hide_index=True)
    metric_choice = st.selectbox("Compare metric", ["F1 Score","Recall","Precision","Accuracy","ROC AUC"])
    chart = px.bar(results.sort_values(metric_choice, ascending=True), x=metric_choice, y="Model", orientation="h",
                   color="Model", title=f"{metric_choice} by model", color_discrete_sequence=["#8d68d7","#b78bff","#d6a5ff","#7654b7"])
    chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False, height=330, margin=dict(l=10,r=10,t=55,b=10), xaxis_range=[0,1])
    st.plotly_chart(chart, use_container_width=True)

    st.markdown("### Confusion matrix · selected model")
    cm = confusion_matrix(y_test, best_predictions, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["Actual normal","Actual spike"], columns=["Predicted normal","Predicted spike"])
    st.dataframe(cm_df, use_container_width=True)
    report = classification_report(y_test, best_predictions, target_names=["Normal","Spike"], zero_division=0, output_dict=True)
    st.markdown("### Error slices")
    err_df = test[[ "hour", "is_weekend", TARGET ]].copy()
    err_df["prediction"] = best_predictions
    err_df["error"] = (err_df[TARGET] != err_df["prediction"]).astype(int)
    err_df["time_period"] = pd.cut(err_df["hour"], bins=[-1,5,11,17,23], labels=["Night","Morning","Afternoon","Evening"])
    slice1 = err_df.groupby("time_period", observed=False)["error"].mean().reset_index(name="error_rate")
    slice2 = err_df.groupby("is_weekend")["error"].mean().reset_index(name="error_rate")
    slice2["group"] = slice2["is_weekend"].map({0:"Weekday",1:"Weekend"})
    s1,s2 = st.columns(2)
    with s1:
        st.markdown("**Error rate by time period**")
        st.dataframe(slice1.style.format({"error_rate":"{:.1%}"}), use_container_width=True, hide_index=True)
    with s2:
        st.markdown("**Error rate by day type**")
        st.dataframe(slice2[["group","error_rate"]].style.format({"error_rate":"{:.1%}"}), use_container_width=True, hide_index=True)
    st.caption(f"Selected model: {best_name}. Model selection uses holdout F1 score for this demonstration; discuss precision/recall trade-offs with your mentor.")

elif page == "Dataset Explorer":
    st.subheader("Dataset explorer")
    st.caption("Inspect schema, data quality, target balance and sample records.")
    x1,x2,x3 = st.columns(3)
    x1.metric("Rows", f"{len(data):,}")
    x2.metric("Predictor columns", len(FEATURES))
    x3.metric("Missing cells", f"{int(data[FEATURES+[TARGET]].isna().sum().sum()):,}")
    st.markdown("### Data preview")
    st.dataframe(data.tail(100), use_container_width=True, hide_index=True)
    st.markdown("### Data dictionary")
    dictionary = pd.DataFrame([
        ["timestamp","Hourly timestamp for the observation","Date/time"],
        ["hour","Hour of day (0–23)","Predictor"],
        ["day_of_week","Day index (Monday=0)","Predictor"],
        ["month","Calendar month (1–12)","Predictor"],
        ["is_weekend","1 if Saturday/Sunday, otherwise 0","Predictor"],
        ["temperature_c","Synthetic temperature in Celsius","Predictor"],
        ["humidity_pct","Synthetic relative humidity percentage","Predictor"],
        ["demand_lag_1_mw","Current-hour demand in MW","Predictor"],
        ["demand_rolling_3h_mw","Mean demand over current and previous 2 hours","Predictor"],
        ["demand_rolling_24h_mw","Mean demand over current and previous 23 hours","Predictor"],
        ["spike_next_hour","1 if next-hour demand exceeds current rolling 24h mean by 48 MW","Target"]
    ], columns=["Column","Meaning","Role"])
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
    st.markdown("### Data quality checks")
    dupes = int(data.duplicated().sum())
    missing = int(data[FEATURES+[TARGET]].isna().sum().sum())
    st.write(f"- Duplicate rows: **{dupes}**")
    st.write(f"- Missing values in model columns: **{missing}**")
    st.write(f"- Target classes: **{sorted(data[TARGET].unique().tolist())}**")
    st.download_button("⬇ Download dataset CSV", data=data.to_csv(index=False).encode("utf-8"),
                       file_name="electricity_demand.csv", mime="text/csv")

st.markdown("---")
st.caption("VOLTIQ · Team 11 · Educational ML prototype · Synthetic demo data — mentor approval and a legitimate dataset are required for final evaluation.")
