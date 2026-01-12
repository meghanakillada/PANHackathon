import os
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from google import genai
from data_generator import generate_multi_source_data

st.set_page_config(page_title="Health Aggregator", layout="wide")

### SESSION STATE MANAGEMENT ###
def init_state():
    st.session_state.setdefault("data_version", 0)
    st.session_state["selected_pair"] = None
    st.session_state.setdefault("ai_insights", None)
    st.session_state.setdefault("ai_insights_version", -1)

def bump_data_version():
    st.session_state.data_version += 1
    st.session_state.selected_pair = None
    st.session_state.ai_insights = None
    st.session_state.ai_insights_version = -1

init_state()

### GEMINI CLIENT SETUP ###
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

### DATA & ANOMALIES & CORRELATIONS ###
def get_unified_data():
    if not os.path.exists("heart_rate.csv"):
        generate_multi_source_data()
    df_hr = pd.read_csv("heart_rate.csv", parse_dates=['date'])
    df_act = pd.read_csv("activity.csv", parse_dates=['date'])
    df_slp = pd.read_csv("sleep.csv", parse_dates=['date'])
    df_nut = pd.read_csv("nutrition.csv", parse_dates=['date'])
    
    df = df_hr.merge(df_act, on='date').merge(df_slp, on='date').merge(df_nut, on='date')
    
    df["prev_day_sleep"] = df["sleep_hrs"].shift(1)
    df["prev_day_caffeine"] = df["caffeine_mg"].shift(1)
    recovery = (df["prev_day_sleep"] * 10) - (df["prev_day_caffeine"] * 0.02) + np.random.normal(10, 5, len(df))
    df["recovery_score"] = np.clip(recovery, 20, 100)

    return df.dropna()

df = get_unified_data()

def get_anomalies(data):
    data = data.copy()
    features = ['heart_rate_bpm', 'sleep_hrs', 'caffeine_mg', 'active_min']
    model = IsolationForest(contamination=0.05, random_state=42)
    data['is_anomaly'] = model.fit_predict(data[features])
    return data[data['is_anomaly'] == -1]

anomalies = get_anomalies(df)

def top_correlation_pairs(df: pd.DataFrame, n: int = 6, skip: int = 3):
    corr = df.select_dtypes(include=[np.number]).corr()

    # Keep upper triangle only (avoid self-corr)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_ut = corr.where(mask)

    pairs = (
        corr_ut.stack()
        .reset_index()
        .rename(columns={"level_0": "metric_a", "level_1": "metric_b", 0: "r"})
    )

    pairs["abs_r"] = pairs["r"].abs()
    pairs = pairs.sort_values("abs_r", ascending=False)

    # Skip top correlations (derived) then take next n
    pairs = pairs.iloc[skip: skip + n]
    return pairs[["metric_a", "metric_b", "r"]]

def plot_metric_pair(df: pd.DataFrame, metric_a: str, metric_b: str):
    # Time series chart
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df["date"], y=df[metric_a], name=metric_a, mode="lines"))
    fig_ts.add_trace(go.Scatter(x=df["date"], y=df[metric_b], name=metric_b, mode="lines"))
    fig_ts.update_layout(
        title=f"Time Series: {metric_a} vs {metric_b}",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Metric",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Scatter plot
    fig_scatter = px.scatter(
        df, x=metric_a, y=metric_b,
        trendline="ols",
        title=f"Relationship: {metric_a} vs {metric_b} (scatter + trendline)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

### LLM ###
def generate_ai_insights(df: pd.DataFrame, anomalies: pd.DataFrame) -> str:
    corr = df.select_dtypes(include=[np.number]).corr().unstack().sort_values(ascending=False)
    top_corr = corr[corr < 1.0].head(3).to_string()

    prompt = f"""You are a professional health coach. Generate 3-5 bullet insights
(at least one explaining anomalies and at least one explaining correlations) and 2 suggested actions.
Use supportive, concise, non-medical language and include caveats where appropriate.

Summary Stats:
{df.describe().to_string()}

Top Correlations:
{top_corr}

Recent Anomalies:
{anomalies.tail(3).to_string()}
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

### DASHBOARD UI ###
st.title("❤️ Personal Health & Wellness Aggregator‍")

### AI ASSISTANT (2 MODES) ###
if not client:
    st.warning("AI features disabled: set GEMINI_API_KEY to enable Gemini insights.")
else:
    tab1, tab2 = st.tabs(["AI Insights", " AI Assistant"])

    # Mode 1: Insights Generator
    with tab1:
        st.subheader("Weekly AI Insights")

        # Auto-generate once per data_version
        if st.session_state.ai_insights is None or st.session_state.ai_insights_version != st.session_state.data_version:
            with st.spinner("Generating insights..."):
                st.session_state.ai_insights = generate_ai_insights(df, anomalies)
                st.session_state.ai_insights_version = st.session_state.data_version

        st.write(st.session_state.ai_insights)
    
    # Mode 2: AI Health Coach Chat
    with tab2:
        user_query = st.text_input("**Chat with your AI Health Coach**", placeholder="Why did my heart rate spike?", label_visibility="visible", )
        if user_query:
            # Simple RAG: find anomalies and co-occurring metric changes
            context = f"Context: You are a professional health coach. Explain to the user in a supportive, actionable, and concise sentences. On dates with spikes, caffeine was {anomalies['caffeine_mg'].mean()} mg vs normal {df['caffeine_mg'].mean()} mg."
            full_prompt = f"{context}\nQuestion: {user_query}\nAnswer grounded in stats:"
        
            response = client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
            st.write(response.text)

# Display KPI Row 
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Avg Sleep (Week)", f"{df['sleep_hrs'].tail(7).mean():.1f} hrs")
kpi2.metric("Avg Heart Rate", f"{df['heart_rate_bpm'].mean():.0f} bpm")
kpi3.metric("Active Minutes", f"{df['active_min'].mean():.0f} min")

# Generate New Data Button
if st.sidebar.button("Generate New Data"):
    generate_multi_source_data()
    bump_data_version()
    st.rerun()

# Chart Settings (Metric Selection)
metric_presets = {
    "Steps View": ["steps"],
    "Stress View": ["heart_rate_bpm", "caffeine_mg"],
    "Habit View": ["screen_hrs", "sleep_hrs"],
    "Training View": ["active_min", "workout_intensity", "recovery_score"],
}

st.sidebar.header("Chart Settings")

st.session_state.setdefault("show_steps", True)
st.session_state.setdefault("show_stress", False)
st.session_state.setdefault("show_habit", False)
st.session_state.setdefault("show_training", False)

show_steps = st.sidebar.checkbox("Steps View", key="show_steps")
show_stress = st.sidebar.checkbox("Stress View", key="show_stress")
show_habit = st.sidebar.checkbox("Habit View", key="show_habit")
show_training = st.sidebar.checkbox("Training View", key="show_training")


selected_metrics = []
if show_steps:
    selected_metrics += metric_presets["Steps View"]
if show_habit:
    selected_metrics += metric_presets["Habit View"]
if show_stress:
    selected_metrics += metric_presets["Stress View"]
if show_training:
    selected_metrics += metric_presets["Training View"]

selected_metrics = list(dict.fromkeys(selected_metrics))

# fallback if user unchecks everything
if not selected_metrics:
    selected_metrics = ["steps"]

# Main Chart
fig = go.Figure()
for metric in selected_metrics:
    fig.add_trace(go.Scatter(x=df['date'], y=df[metric], name=metric, mode='lines'))

# Label Anomalies
if show_stress:
    fig.add_trace(go.Scatter(
        x=anomalies["date"],
        y=anomalies["heart_rate_bpm"],
        mode="markers",
        name="Anomaly Detected",
        marker=dict(color="red", size=10, symbol="x"),
    ))

st.plotly_chart(fig, use_container_width=True)

# List Anomalies
with st.expander("Detailed Anomaly List"):
    st.table(anomalies[['date', 'heart_rate_bpm', 'caffeine_mg', 'sleep_hrs']].tail())

# Recovery Simulator
st.sidebar.markdown("---")
st.sidebar.header("🚀 Recovery Simulator")
st.sidebar.write("Predict tomorrow's recovery based on tonight's plan:")
sim_sleep = st.sidebar.slider("Sleep Hours", 4.0, 10.0, 8.0)
sim_caffeine = st.sidebar.slider("Caffeine (mg)", 0, 600, 100)

# train linear regression model for the simulator
X = df[['prev_day_sleep', 'prev_day_caffeine']]
y = df['recovery_score']
model = LinearRegression().fit(X, y)
predicted_recovery = model.predict([[sim_sleep, sim_caffeine]])[0]

st.sidebar.metric("Predicted Recovery Score", f"{predicted_recovery:.1f}/100")

# List Correlations (Buttons)
pairs = top_correlation_pairs(df, n=6, skip=3)

with st.expander("Top Correlations List"):
    cols = st.columns(2)
    for i, (_, row) in enumerate(pairs.iterrows()):
        a, b, r = row["metric_a"], row["metric_b"], row["r"]
        label = f"{a} ↔ {b} (r={r:+.2f})"
        with cols[i % 2]:
            if st.button(label, key=f"pair_{a}_{b}"):
                st.session_state.selected_pair = (a, b, r)

# Correlation Pair Plot
if st.session_state.selected_pair:
    st.subheader("📈 Selected Correlation Plot")
    a, b, r = st.session_state.selected_pair
    st.caption(f"Selected: **{a}** vs **{b}** (Pearson r = {r:+.2f})")
    plot_metric_pair(df, a, b)
else:
    st.info("Click a correlation pair above in the dropdown to visualize it.")