import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Public Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------
st.markdown("""
<style>
body { background-color: #f4f6f9; }

.app-header {
    background: linear-gradient(90deg, #1f4fd8, #3f8efc);
    padding: 22px;
    border-radius: 14px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}
.app-header h1 { margin: 0; font-size: 32px; }
.app-header p { margin-top: 6px; opacity: 0.9; }

.card {
    background-color: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    border-left: 6px solid #3f8efc;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    text-align: center;
}
.metric-title { font-size: 14px; color: #6b7280; }
.metric-value { font-size: 26px; font-weight: bold; color: #1f4fd8; }

.chat-box {
    background-color: #0f172a;
    color: #e5e7eb;
    padding: 18px;
    border-radius: 14px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>🛡️ AI Public Fraud Detection System</h1>
    <p>Machine Learning–based anomaly detection for government audits</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("🔍 Controls")
st.sidebar.markdown("""
**Project:** AI Fraud Detection  
**Domain:** Public Finance  
**Built with:** ML + Streamlit  
""")

uploaded_file = st.sidebar.file_uploader("📂 Upload Transaction CSV", type="csv")

if uploaded_file is None:
    st.info("👈 Upload a CSV file to start analysis")
    st.stop()

# ---------------------------------------------------
# LOAD & VALIDATE DATA
# ---------------------------------------------------
df = pd.read_csv(uploaded_file)

required_cols = ["transaction_id","department_id","vendor_id","amount","transaction_time"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------
df["transaction_time"] = pd.to_datetime(df["transaction_time"])
df["hour"] = df["transaction_time"].dt.hour
df["is_night"] = df["hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)
df["is_weekend"] = df["transaction_time"].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

df["log_amount"] = np.log1p(df["amount"])
dept_mean = df.groupby("department_id")["amount"].transform("mean")
dept_std = df.groupby("department_id")["amount"].transform("std")

df["amount_zscore_dept"] = (df["amount"] - dept_mean) / dept_std
df["amount_vs_dept_mean"] = df["amount"] / dept_mean
df["vendor_txn_count"] = df.groupby("vendor_id")["vendor_id"].transform("count")
df["vendor_avg_amount"] = df.groupby("vendor_id")["amount"].transform("mean")
df["vendor_amount_ratio"] = df["amount"] / df["vendor_avg_amount"]

features = [
    "log_amount","amount_zscore_dept","amount_vs_dept_mean",
    "hour","is_night","is_weekend",
    "vendor_txn_count","vendor_avg_amount","vendor_amount_ratio"
]

X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
X_scaled = scaler.transform(X)

# ---------------------------------------------------
# MODEL PREDICTION
# ---------------------------------------------------
df["anomaly_score"] = model.decision_function(X_scaled)
df["fraud_flag"] = model.predict(X_scaled)
df["fraud_flag"] = df["fraud_flag"].apply(lambda x: 1 if x == -1 else 0)

df["risk_score"] = (
    (df["anomaly_score"].max() - df["anomaly_score"]) /
    (df["anomaly_score"].max() - df["anomaly_score"].min())
) * 100

df["risk_level"] = pd.cut(
    df["risk_score"], bins=[0,30,70,100],
    labels=["Low","Medium","High"]
)

# ---------------------------------------------------
# EXPLANATIONS
# ---------------------------------------------------
def explain(row):
    reasons = []
    if row["amount_zscore_dept"] > 3: reasons.append("High amount vs department")
    if row["vendor_amount_ratio"] > 3: reasons.append("Vendor spike")
    if row["is_night"]: reasons.append("Night transaction")
    if row["is_weekend"]: reasons.append("Weekend transaction")
    if row["vendor_txn_count"] > 50: reasons.append("High vendor frequency")
    return ", ".join(reasons) if reasons else "Normal pattern"

df["explanation"] = df.apply(explain, axis=1)

# ---------------------------------------------------
# FILTERS
# ---------------------------------------------------
dept_filter = st.sidebar.multiselect("Department", df["department_id"].unique(), df["department_id"].unique())
vendor_filter = st.sidebar.multiselect("Vendor", df["vendor_id"].unique(), df["vendor_id"].unique())

df = df[
    (df["department_id"].isin(dept_filter)) &
    (df["vendor_id"].isin(vendor_filter))
]

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analysis", "📄 Audits", "💬 Chatbot"])

# -------------------- OVERVIEW --------------------
with tab1:
    st.markdown("<div class='card'><h3>📊 Key Metrics</h3></div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)

    c1.markdown(f"<div class='metric-card'><div class='metric-title'>Total Transactions</div><div class='metric-value'>{len(df)}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-title'>Fraud Detected</div><div class='metric-value'>{df['fraud_flag'].sum()}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-title'>High Risk</div><div class='metric-value'>{(df['risk_level']=='High').sum()}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-title'>Fraud %</div><div class='metric-value'>{round(df['fraud_flag'].mean()*100,2)}%</div></div>", unsafe_allow_html=True)

# -------------------- ANALYSIS --------------------
with tab2:
    st.markdown("<div class='card'><h3>📈 Risk Distribution</h3></div>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.hist(df["risk_score"], bins=30)
    st.pyplot(fig)

    st.markdown("<div class='card'><h3>🏢 Department Risk</h3></div>", unsafe_allow_html=True)
    st.bar_chart(df.groupby("department_id")["risk_score"].mean())

# -------------------- AUDITS --------------------
with tab3:
    flagged = df[df["fraud_flag"]==1]
    st.markdown("<div class='card'><h3>📄 Flagged Transactions</h3></div>", unsafe_allow_html=True)
    st.dataframe(flagged)

    csv = flagged.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Audit Report", csv, "audit_report.csv")

# -------------------- CHATBOT --------------------
with tab4:
    st.markdown("<div class='card'><h3>💬 Auditor Assistant</h3></div>", unsafe_allow_html=True)
    q = st.text_input("Ask a question")
    if q:
        if "high risk" in q.lower():
            answer = f"High-risk cases: {(df['risk_level']=='High').sum()}"
        elif "top vendor" in q.lower():
            answer = df[df["fraud_flag"]==1]["vendor_id"].value_counts().head(5).to_string()
        else:
            answer = "Ask about high-risk cases or vendors."

        st.markdown(f"<div class='chat-box'>🤖 {answer}</div>", unsafe_allow_html=True)
