# Personal Health & Wellness Aggregator (Streamlit MVP)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

_____## Ingestion options (MVP)
1) **Apple Health export**: upload either:
   - `export.xml` (from Apple Health export), or
   - `export.zip` containing `apple_health_export/export.xml` or `export.xml`

2) **Synthetic CSVs** (included in `data/synthetic/`): heart rate, activity, sleep, nutrition

## Notes
- All ingested data is normalized into a canonical event table, then aggregated into a daily feature store.
- LLM features are optional; set `OPENAI_API_KEY` in your environment to enable._____


######3
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from google import genai
from data_generator import generate_multi_source_data

st.set_page_config(page_title="Health Aggregator", layout="wide")
client = genai.Client(api_key="AIzaSyAQXWKWQtGIFkrCGmDWt2ZQhKzwRP6Hwhw")

# --- DATA & ANOMALIES ---
def get_unified_data():
    if not os.path.exists("heart_rate.csv"):
        generate_multi_source_data()
    
    # Load all sources
    df_hr = pd.read_csv("heart_rate.csv", parse_dates=['date'])
    df_act = pd.read_csv("activity.csv", parse_dates=['date'])
    df_slp = pd.read_csv("sleep.csv", parse_dates=['date'])
    df_nut = pd.read_csv("nutrition.csv", parse_dates=['date'])
    
    # Merge all on date
    unified = df_hr.merge(df_act, on='date').merge(df_slp, on='date').merge(df_nut, on='date')
    unified.to_csv("unified_health_data.csv", index=False)
    return unified

df = get_unified_data()

def get_anomalies(data):
    features = ['heart_rate_bpm', 'sleep_hrs', 'caffeine_mg', 'active_min']
    model = IsolationForest(contamination=0.05, random_state=42)
    data['is_anomaly'] = model.fit_predict(data[features])
    return data[data['is_anomaly'] == -1]

anomalies = get_anomalies(df)

# --- UI: DASHBOARD ---
st.title("❤️‍ Personal Health Wellness Aggregator")

# KPI Row 
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Avg Sleep (Week)", f"{df['sleep_hrs'].tail(7).mean():.1f} hrs") # [cite: 29]
kpi2.metric("Avg Heart Rate", f"{df['heart_rate_bpm'].mean():.0f} bpm") # [cite: 31]
kpi3.metric("Active Minutes", f"{df['active_min'].mean():.0f} min") # [cite: 30]

# Generate Button [cite: 23]
if st.sidebar.button("Generate New Data"):
    generate_multi_source_data()
    st.rerun()

# Unified Chart with Toggles 
st.subheader("Unified Health Timeline")
metrics = ['heart_rate_bpm', 'sleep_hrs', 'active_min', 'caffeine_mg', 'sugar_g']
selected_metrics = [m for m in metrics if st.sidebar.checkbox(f"Show {m}", value=True)]

fig = go.Figure()
colors = px.colors.qualitative.Plotly
for i, m in enumerate(selected_metrics):
    fig.add_trace(go.Scatter(x=df['date'], y=df[m], name=m, line=dict(color=colors[i % len(colors)])))

# Label Anomalies on Chart 
fig.add_trace(go.Scatter(
    x=anomalies['date'], y=anomalies['heart_rate_bpm'],
    mode='markers', name='Anomaly Detected',
    marker=dict(color='red', size=10, symbol='x')
))
st.plotly_chart(fig, use_container_width=True)

# List Anomalies [cite: 33]
with st.expander("Detailed Anomaly List"):
    st.table(anomalies[['date', 'heart_rate_bpm', 'caffeine_mg', 'sleep_hrs']].tail())

# --- AI ASSISTANT (2 MODES) [cite: 46] ---
st.divider()
tab1, tab2 = st.tabs(["Insights Generator", "Health Q&A"])

# Mode 1: Insights Generator [cite: 47]
with tab1:
    if st.button("Generate Weekly Insights"):
        with st.spinner("Gemini is analyzing your health patterns..."):
            # Passing summary stats, anomalies, and correlations [cite: 42, 43, 44]
            corr = df.corr().unstack().sort_values(ascending=False)
            top_corr = corr[corr < 1.0].head(2).to_string()
            
            prompt = f"""You are a professional health coach. Generate 3-5 bullet insights and 2 suggested actions for the user based on the user's health data. Explain to the user in a supportive, actionable, and concise sentences. Use non-medical language and include caveats where appropriate.
            Summary Stats: {df.describe().to_string()}
            Top Correlations: {top_corr}
            Recent Anomalies: {anomalies.tail(3).to_string()}
            """ # [cite: 49, 50]
            
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            st.write(response.text)

# Mode 2: Q&A Assistant 
with tab2:
    user_query = st.text_input("Ask about your health (e.g., 'Why did my heart rate spike?')") # [cite: 55]
    if user_query:
        # Simple RAG: find anomalies and co-occurring metric changes [cite: 52, 57, 58]
        context = f"Context: On dates with spikes, caffeine was {anomalies['caffeine_mg'].mean()} mg vs normal {df['caffeine_mg'].mean()} mg."
        full_prompt = f"{context}\nQuestion: {user_query}\nAnswer grounded in stats:" # [cite: 38]
        
        response = client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
        st.write(response.text) # [cite: 56, 59]