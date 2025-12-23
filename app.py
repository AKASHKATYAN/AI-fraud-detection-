import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# ----------------------------------
# BASIC PAGE SETUP
# ----------------------------------
st.set_page_config(page_title="AI Fraud Detection", layout="wide")
st.title("AI-Based Public Fraud Detection System")

# ----------------------------------
# CHECK MODEL FILES
# ----------------------------------
if not os.path.exists("fraud_model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("❌ Model or scaler file not found. Please ensure fraud_model.pkl and scaler.pkl are present.")
    st.stop()

# Load model and scaler
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------------
# FILE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Government Transactions CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("⬆️ Please upload a CSV file to generate the fraud dashboard.")
    st.stop()

# ----------------------------------
# LOAD DATA
# ----------------------------------
df = pd.read_csv(uploaded_file)

st.success("✅ File uploaded successfully!")

# ----------------------------------
# REQUIRED COLUMN VALIDATION
# ----------------------------------
required_cols = [
    "transaction_id",
    "department_id",
    "vendor_id",
    "amount",
    "transaction_time"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

# ----------------------------------
# FEATURE ENGINEERING
# ----------------------------------
df["log_amount"] = np.log1p(df["amount"])

dept_mean = df.groupby("department_id")["amount"].transform("mean")
dept_std = df.groupby("department_id")["amount"].transform("std")

df["amount_zscore_dept"] = (df["amount"] - dept_mean) / dept_std
df["amount_vs_dept_mean"] = df["amount"] / dept_mean

df["transaction_time"] = pd.to_datetime(df["transaction_time"])
df["hour"] = df["transaction_time"].dt.hour
df["is_night"] = df["hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)
df["is_weekend"] = df["transaction_time"].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

df["vendor_txn_count"] = df.groupby("vendor_id")["vendor_id"].transform("count")
df["vendor_avg_amount"] = df.groupby("vendor_id")["amount"].transform("mean")
df["vendor_amount_ratio"] = df["amount"] / df["vendor_avg_amount"]

features = [
    "log_amount",
    "amount_zscore_dept",
    "amount_vs_dept_mean",
    "hour",
    "is_night",
    "is_weekend",
    "vendor_txn_count",
    "vendor_avg_amount",
    "vendor_amount_ratio"
]

X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
X_scaled = scaler.transform(X)

# ----------------------------------
# MODEL PREDICTION
# ----------------------------------
df["anomaly_score"] = model.decision_function(X_scaled)
df["fraud_predicted"] = model.predict(X_scaled)
df["fraud_predicted"] = df["fraud_predicted"].apply(lambda x: 1 if x == -1 else 0)

# ----------------------------------
# DASHBOARD METRICS
# ----------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df))
col2.metric("Fraud Cases Detected", int(df["fraud_predicted"].sum()))
col3.metric("Fraud Rate (%)", round(df["fraud_predicted"].mean() * 100, 2))

# ----------------------------------
# FRAUD SCORE DISTRIBUTION
# ----------------------------------
st.subheader("📉 Fraud Risk Score Distribution")

fig1, ax1 = plt.subplots()
ax1.hist(df["anomaly_score"], bins=50)
ax1.set_xlabel("Anomaly Score")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# ----------------------------------
# TOP RISKY VENDORS
# ----------------------------------
st.subheader("🏭 Top Risky Vendors")

vendor_fraud = (
    df[df["fraud_predicted"] == 1]["vendor_id"]
    .value_counts()
    .head(10)
)

if vendor_fraud.empty:
    st.info("No high-risk vendors detected.")
else:
    st.bar_chart(vendor_fraud)

# ----------------------------------
# FLAGGED TRANSACTIONS TABLE
# ----------------------------------
st.subheader("🚩 Flagged Transactions (Audit List)")

flagged_df = df[df["fraud_predicted"] == 1][
    ["transaction_id", "department_id", "vendor_id", "amount", "anomaly_score"]
]

if flagged_df.empty:
    st.info("No suspicious transactions found.")
else:
    st.dataframe(flagged_df)
