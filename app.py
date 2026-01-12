import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ChitraGuptAI",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# ENHANCED GLASSMORPHISM STYLES
# --------------------------------------------------
st.markdown("""
<style>
/* ========== GLOBAL THEME ========== */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
}

/* Main content area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ========== NAVIGATION BUTTON ========== */
.nav-button {
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 1000;
}

.back-btn {
    background: rgba(30, 58, 138, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #60a5fa;
    padding: 10px 20px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.back-btn:hover {
    background: rgba(30, 58, 138, 0.8);
    border-color: rgba(59, 130, 246, 0.6);
    transform: translateX(-3px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}

/* ========== HEADER STYLES ========== */
.main-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 2rem;
    background: rgba(15, 23, 42, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

.big-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    font-weight: 400;
}

/* ========== GLASS CARDS ========== */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(71, 85, 105, 0.4);
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(59, 130, 246, 0.5);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
    transform: translateY(-2px);
}

/* ========== METRIC BOXES ========== */
[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
}

[data-testid="stMetricLabel"] {
    font-size: 0.9rem;
    color: #cbd5e1;
    font-weight: 500;
}

.stMetric {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 1.2rem;
    border: 1px solid rgba(71, 85, 105, 0.4);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.3s ease;
}

.stMetric:hover {
    border-color: rgba(59, 130, 246, 0.6);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.3);
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid rgba(71, 85, 105, 0.4);
}

[data-testid="stSidebar"] .stMarkdown {
    color: #e2e8f0;
}

/* Sidebar widgets */
.stSelectbox, .stMultiSelect {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 10px;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.5);
    padding: 0.5rem;
    border-radius: 12px;
    border: 1px solid rgba(71, 85, 105, 0.3);
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.4);
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    border: 1px solid transparent;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: rgba(59, 130, 246, 0.3);
    color: #60a5fa;
    border-color: rgba(59, 130, 246, 0.5);
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(59, 130, 246, 0.2);
    color: #93c5fd;
}

/* ========== DATAFRAMES ========== */
[data-testid="stDataFrame"] {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    border: 1px solid rgba(71, 85, 105, 0.4);
    overflow: hidden;
}

/* DataFrame headers */
.stDataFrame thead tr th {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #60a5fa !important;
    font-weight: 600 !important;
    border-bottom: 2px solid rgba(59, 130, 246, 0.5) !important;
}

/* DataFrame rows */
.stDataFrame tbody tr {
    background: rgba(15, 23, 42, 0.4);
    transition: all 0.2s ease;
}

.stDataFrame tbody tr:hover {
    background: rgba(59, 130, 246, 0.15) !important;
    transform: scale(1.01);
}

/* ========== CHARTS ========== */
.stPlotlyChart, [data-testid="stImage"] {
    background: rgba(15, 23, 42, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid rgba(71, 85, 105, 0.3);
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    border: 1px solid rgba(59, 130, 246, 0.5);
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #60a5fa 100%);
    border-color: rgba(59, 130, 246, 0.8);
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(59, 130, 246, 0.5);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    border-radius: 10px;
    font-weight: 600;
}

/* ========== CHAT INTERFACE ========== */
.chat-container {
    background: rgba(15, 23, 42, 0.5);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(71, 85, 105, 0.4);
    padding: 1.5rem;
    margin-top: 1rem;
}

.user-message {
    background: rgba(59, 130, 246, 0.2);
    border-left: 4px solid #3b82f6;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    color: #e2e8f0;
}

.ai-message {
    background: rgba(16, 185, 129, 0.15);
    border-left: 4px solid #10b981;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    color: #e2e8f0;
}

.question-btn {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(71, 85, 105, 0.4);
    color: #94a3b8;
    padding: 0.8rem;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 0.3rem 0;
}

.question-btn:hover {
    background: rgba(59, 130, 246, 0.2);
    border-color: rgba(59, 130, 246, 0.5);
    color: #60a5fa;
}

/* ========== ALERTS ========== */
.stAlert {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
}

/* Success */
[data-baseweb="notification"][kind="positive"] {
    background: rgba(16, 185, 129, 0.2);
    border-left-color: #10b981;
}

/* Warning */
[data-baseweb="notification"][kind="warning"] {
    background: rgba(245, 158, 11, 0.2);
    border-left-color: #f59e0b;
}

/* Info */
[data-baseweb="notification"][kind="info"] {
    background: rgba(59, 130, 246, 0.2);
    border-left-color: #3b82f6;
}

/* ========== INPUTS ========== */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(30, 41, 59, 0.6);
    color: #e2e8f0;
    border: 1px solid rgba(71, 85, 105, 0.4);
    border-radius: 8px;
}

/* ========== FILE UPLOADER ========== */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    border: 2px dashed rgba(71, 85, 105, 0.4);
    padding: 1rem;
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.5);
}

::-webkit-scrollbar-thumb {
    background: rgba(71, 85, 105, 0.6);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(100, 116, 139, 0.8);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION - Back to Main Website Button
# --------------------------------------------------
MAIN_WEBSITE_URL = "https://auditnova.lovable.app/"  # 🔧 CONFIGURE THIS URL

st.markdown(f"""
<div class="nav-button">
    <a href="{MAIN_WEBSITE_URL}" class="back-btn" target="_self">
        ← Back to Main Website
    </a>
</div>
""", unsafe_allow_html=True)

# Add spacing for the fixed button
st.markdown("<br><br>", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="big-title">🛡️ AI-Based Public Fraud Detection System</div>
    <p class="subtitle">Detect anomalies, assess risk, and support government audits using AI</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------
# SYNTHETIC DATA GENERATOR (LARGE DATASET)
# --------------------------------------------------
@st.cache_data
def generate_sample_data(n=5000):
    departments = ["Health", "Education", "Transport", "Defense", "Rural", "Urban"]
    vendors = [f"V{100+i}" for i in range(40)]
    base_time = datetime.now() - timedelta(days=90)

    data = []
    for i in range(n):
        dept = random.choice(departments)
        vendor = random.choice(vendors)

        amount = np.random.lognormal(mean=10, sigma=1.2)
        if random.random() < 0.08:  # fraud spikes
            amount *= random.randint(5, 20)

        txn_time = base_time + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        data.append([
            100000 + i,
            dept,
            vendor,
            round(amount, 2),
            txn_time
        ])

    return pd.DataFrame(
        data,
        columns=["transaction_id", "department_id", "vendor_id", "amount", "transaction_time"]
    )

# --------------------------------------------------
# SIDEBAR - Data Input Section
# --------------------------------------------------
st.sidebar.header("📂 Data Input")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
use_sample = st.sidebar.button("🧪 Use Sample Dataset")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Uploaded dataset loaded")

elif use_sample:
    df = generate_sample_data()
    st.sidebar.info("📊 Using large sample dataset (5,000 rows)")

else:
    st.info("📤 Upload a CSV or click **Use Sample Dataset** to explore the dashboard.")
    st.stop()

# --------------------------------------------------
# FEATURE ENGINEERING (ML LOGIC UNCHANGED)
# --------------------------------------------------
df["transaction_time"] = pd.to_datetime(df["transaction_time"])
df["hour"] = df["transaction_time"].dt.hour
df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
df["is_weekend"] = (df["transaction_time"].dt.weekday >= 5).astype(int)

df["log_amount"] = np.log1p(df["amount"])

dept_mean = df.groupby("department_id")["amount"].transform("mean")
dept_std = df.groupby("department_id")["amount"].transform("std")

df["amount_zscore_dept"] = (df["amount"] - dept_mean) / dept_std
df["amount_vs_dept_mean"] = df["amount"] / dept_mean

df["vendor_txn_count"] = df.groupby("vendor_id")["vendor_id"].transform("count")
df["vendor_avg_amount"] = df.groupby("vendor_id")["amount"].transform("mean")
df["vendor_amount_ratio"] = df["amount"] / df["vendor_avg_amount"]

features = [
    "log_amount", "amount_zscore_dept", "amount_vs_dept_mean",
    "hour", "is_night", "is_weekend",
    "vendor_txn_count", "vendor_avg_amount", "vendor_amount_ratio"
]

X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
X_scaled = scaler.transform(X)

# --------------------------------------------------
# MODEL PREDICTION (ML LOGIC UNCHANGED)
# --------------------------------------------------
df["anomaly_score"] = model.decision_function(X_scaled)
df["fraud_flag"] = (model.predict(X_scaled) == -1).astype(int)

df["risk_score"] = (
    (df["anomaly_score"].max() - df["anomaly_score"]) /
    (df["anomaly_score"].max() - df["anomaly_score"].min())
) * 100

df["risk_level"] = pd.cut(
    df["risk_score"], bins=[0, 30, 70, 100],
    labels=["Low", "Medium", "High"]
)

# --------------------------------------------------
# EXPLANATIONS (ML LOGIC UNCHANGED)
# --------------------------------------------------
def explain(row):
    reasons = []
    if row["amount_zscore_dept"] > 3:
        reasons.append("Unusual amount for department")
    if row["vendor_amount_ratio"] > 3:
        reasons.append("Vendor billing spike")
    if row["is_night"]:
        reasons.append("Night transaction")
    if row["is_weekend"]:
        reasons.append("Weekend activity")
    if row["vendor_txn_count"] > 80:
        reasons.append("High vendor frequency")
    return ", ".join(reasons) if reasons else "Normal behavior"

df["explanation"] = df.apply(explain, axis=1)

# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.sidebar.header("🔍 Filters")

dept_filter = st.sidebar.multiselect(
    "Department",
    df["department_id"].unique(),
    df["department_id"].unique()
)

df = df[df["department_id"].isin(dept_filter)]

# --------------------------------------------------
# TABS WITH ENHANCED ICONS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview Dashboard",
    "📈 Risk Analysis",
    "🚨 Flagged Audits",
    "💬 AI Assistant"
])

# ---------------- TAB 1: OVERVIEW DASHBOARD ----------------
with tab1:
    st.subheader("📊 Key Performance Indicators")
    
    # KPI Metrics in glass cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Detected", f"{df['fraud_flag'].sum():,}")
    c3.metric("High Risk Cases", f"{(df['risk_level'] == 'High').sum():,}")
    c4.metric("Fraud Rate", f"{round(df['fraud_flag'].mean() * 100, 2)}%")

# ---------------- TAB 2: RISK ANALYSIS ----------------
with tab2:
    st.subheader("📈 Risk Distribution Analysis")
    
    # Risk Score Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.hist(df["risk_score"], bins=40, color='#3b82f6', alpha=0.7, edgecolor='#60a5fa')
    ax.set_xlabel('Risk Score', color='#e2e8f0')
    ax.set_ylabel('Frequency', color='#e2e8f0')
    ax.tick_params(colors='#e2e8f0')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['top'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['right'].set_color('#475569')
    ax.grid(True, alpha=0.2, color='#475569')
    st.pyplot(fig)

    st.subheader("🏢 Department Risk Comparison")
    st.bar_chart(df.groupby("department_id")["risk_score"].mean())

    st.subheader("⏰ Fraud Pattern by Hour")
    st.line_chart(df.groupby("hour")["fraud_flag"].mean())

# ---------------- TAB 3: FLAGGED AUDITS ----------------
with tab3:
    st.subheader("🚨 Suspicious Transactions Requiring Audit")
    
    flagged = df[df["fraud_flag"] == 1][[
        "transaction_id", "department_id", "vendor_id",
        "amount", "risk_score", "risk_level", "explanation"
    ]].sort_values("risk_score", ascending=False)

    if flagged.empty:
        st.success("✅ No suspicious transactions detected in filtered data.")
    else:
        st.warning(f"⚠️ Found {len(flagged)} suspicious transactions requiring immediate review")
        st.dataframe(flagged, use_container_width=True)

        st.download_button(
            "📥 Download Suspicious Transactions CSV",
            flagged.to_csv(index=False),
            "suspicious_transactions.csv",
            "text/csv"
        )

# ---------------- TAB 4: AI AUDITOR ASSISTANT ----------------
with tab4:
    st.markdown("""
    <div class="chat-container">
        <h2 style="color: #60a5fa; margin-bottom: 1rem;">💬 Virtual Auditor Assistant</h2>
        <p style="color: #94a3b8; margin-bottom: 1.5rem;">AI-powered assistant to help auditors quickly understand risks and investigate anomalies</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Quick Audit Questions")

    question = st.selectbox(
        "Select a question to get instant insights:",
        [
            "Select a question...",
            "How many high-risk transactions are there?",
            "What is the total fraud detected?",
            "Who are the top risky vendors?",
            "Which department has highest risk?",
            "Give risk-level summary",
            "Show fraud trend by hour",
            "Explain how this AI model works"
        ]
    )

    st.markdown("### 🤖 AI Response")

    # Chat-like response interface
    if question != "Select a question...":
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 Auditor:</strong> {question}
        </div>
        """, unsafe_allow_html=True)

    if question == "How many high-risk transactions are there?":
        count = (df["risk_level"] == "High").sum()
        st.markdown(f"""
        <div class="ai-message">
            <strong>🤖 AI Assistant:</strong><br>
            🔴 There are <strong>{count} high-risk transactions</strong> requiring immediate audit attention. These cases show significant deviation from normal patterns.
        </div>
        """, unsafe_allow_html=True)

    elif question == "What is the total fraud detected?":
        frauds = df["fraud_flag"].sum()
        st.markdown(f"""
        <div class="ai-message">
            <strong>🤖 AI Assistant:</strong><br>
            🚨 The system detected <strong>{frauds} suspicious transactions</strong> across all departments. These represent {round((frauds/len(df))*100, 2)}% of total transactions analyzed.
        </div>
        """, unsafe_allow_html=True)

    elif question == "Who are the top risky vendors?":
        vendors = df[df["fraud_flag"] == 1]["vendor_id"].value_counts().head(5)
        if vendors.empty:
            st.markdown("""
            <div class="ai-message">
                <strong>🤖 AI Assistant:</strong><br>
                ✅ No risky vendors detected in the current filtered dataset.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ai-message">
                <strong>🤖 AI Assistant:</strong><br>
                🚩 <strong>Top 5 Risky Vendors:</strong>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(vendors.reset_index().rename(columns={
                "index": "Vendor ID",
                "vendor_id": "Fraud Count"
            }), use_container_width=True)

    elif question == "Which department has highest risk?":
        dept = df.groupby("department_id")["risk_score"].mean().idxmax()
        score = df.groupby("department_id")["risk_score"].mean().max()
        st.markdown(f"""
        <div class="ai-message">
            <strong>🤖 AI Assistant:</strong><br>
            🏢 <strong>{dept} Department</strong> has the highest average risk score of <strong>{round(score, 2)}</strong>. Recommend prioritizing audit resources here.
        </div>
        """, unsafe_allow_html=True)

    elif question == "Give risk-level summary":
        summary = df["risk_level"].value_counts()
        st.markdown("""
        <div class="ai-message">
            <strong>🤖 AI Assistant:</strong><br>
            📊 <strong>Risk Distribution Summary:</strong>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(summary.reset_index().rename(columns={
            "index": "Risk Level",
            "risk_level": "Transaction Count"
        }), use_container_width=True)

    elif question == "Show fraud trend by hour":
        st.markdown("""
        <div class="ai-message">
            <strong>🤖 AI Assistant:</strong><br>
            ⏰ <strong>Fraud Pattern by Hour of Day:</strong><br>
            This chart shows the average fraud rate across different hours. Look for spikes during unusual times.
        </div>
        """, unsafe_allow_html=True)
        trend = df.groupby("hour")["fraud_flag"].mean()
        st.line_chart(trend)

    elif question == "Explain how this AI model works":
        st.markdown("""
        <div class="ai-message">
            <strong>🤖 AI Assistant:</strong><br><br>
            <strong>🧠 AI Model Explanation</strong><br><br>
            
            <strong>Algorithm:</strong> Isolation Forest (Unsupervised Anomaly Detection)<br><br>
            
            <strong>How it Works:</strong><br>
            • Learns normal spending behavior automatically without labeled fraud data<br>
            • Isolates anomalies by measuring how different transactions are from typical patterns<br>
            • Analyzes multiple dimensions: transaction amount, timing, vendor behavior, and department patterns<br><br>
            
            <strong>Key Features Used:</strong><br>
            • Transaction amount (log-transformed)<br>
            • Z-score vs department average<br>
            • Time patterns (hour, night, weekend)<br>
            • Vendor transaction frequency and amount ratios<br><br>
            
            <strong>Output:</strong><br>
            • <strong>Risk Score (0-100):</strong> Higher scores indicate greater anomaly<br>
            • <strong>Risk Level:</strong> Low (0-30), Medium (30-70), High (70-100)<br>
            • <strong>Explanation:</strong> Human-readable reasons for flagging<br><br>
            
            <strong>Benefits:</strong><br>
            ✅ No need for historical fraud labels<br>
            ✅ Adapts to each department's spending patterns<br>
            ✅ Prioritizes audits by risk level<br>
            ✅ Provides explainable results for investigators
        </div>
        """, unsafe_allow_html=True)

    elif question == "Select a question...":
        st.info("👆 Please select a question from the dropdown above to get AI-powered insights.")

    # Custom question input
    st.markdown("---")
    st.markdown("### ✍️ Ask Your Own Question")

    custom_q = st.text_input("Type your audit-related question here")

    if custom_q:
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 Auditor:</strong> {custom_q}
        </div>
        <div class="ai-message">
            <strong>🤖 AI Assistant:</strong><br>
            ⚠️ Free-text understanding is limited in this demo. For best results, please use the quick questions dropdown above. For complex queries, consider connecting this system to a large language model API.
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    © 2026 AI Public Fraud Detection
</div>
""")
