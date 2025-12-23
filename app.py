import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(page_title="AI Public Fraud Detection", layout="wide")
st.title("🛡️ AI-Based Public Fraud Detection System")

# -------------------------------------
# LOAD MODEL & SCALER
# -------------------------------------
@st.cache_resource
def load_artifacts():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# -------------------------------------
# FILE UPLOAD
# -------------------------------------
uploaded_file = st.file_uploader("📂 Upload Transaction CSV", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV file to start fraud analysis")
    st.stop()

df = pd.read_csv(uploaded_file)

# -------------------------------------
# VALIDATION
# -------------------------------------
required_cols = ["transaction_id","department_id","vendor_id","amount","transaction_time"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

# -------------------------------------
# FEATURE ENGINEERING
# -------------------------------------
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

X = df[features].replace([np.inf,-np.inf],0).fillna(0)
X_scaled = scaler.transform(X)

# -------------------------------------
# MODEL PREDICTION
# -------------------------------------
df["anomaly_score"] = model.decision_function(X_scaled)
df["fraud_flag"] = model.predict(X_scaled)
df["fraud_flag"] = df["fraud_flag"].apply(lambda x: 1 if x == -1 else 0)

# -------------------------------------
# RISK SCORING
# -------------------------------------
df["risk_score"] = ((df["anomaly_score"].max() - df["anomaly_score"]) /
                    (df["anomaly_score"].max() - df["anomaly_score"].min())) * 100

df["risk_level"] = pd.cut(
    df["risk_score"],
    bins=[0,30,70,100],
    labels=["Low","Medium","High"]
)

# -------------------------------------
# EXPLAINABLE AI
# -------------------------------------
def explain(row):
    reasons = []
    if row["amount_zscore_dept"] > 3:
        reasons.append("Unusually high amount for department")
    if row["vendor_amount_ratio"] > 3:
        reasons.append("Vendor amount spike")
    if row["is_night"] == 1:
        reasons.append("Night-time transaction")
    if row["is_weekend"] == 1:
        reasons.append("Weekend transaction")
    if row["vendor_txn_count"] > 50:
        reasons.append("High frequency vendor")
    return ", ".join(reasons) if reasons else "Normal pattern"

df["explanation"] = df.apply(explain, axis=1)

# -------------------------------------
# FILTERS
# -------------------------------------
st.sidebar.header("🔍 Filters")

dept_filter = st.sidebar.multiselect(
    "Department",
    options=df["department_id"].unique(),
    default=df["department_id"].unique()
)

vendor_filter = st.sidebar.multiselect(
    "Vendor",
    options=df["vendor_id"].unique(),
    default=df["vendor_id"].unique()
)

amount_range = st.sidebar.slider(
    "Amount Range",
    int(df["amount"].min()),
    int(df["amount"].max()),
    (int(df["amount"].min()), int(df["amount"].max()))
)

df = df[
    (df["department_id"].isin(dept_filter)) &
    (df["vendor_id"].isin(vendor_filter)) &
    (df["amount"] >= amount_range[0]) &
    (df["amount"] <= amount_range[1])
]

# -------------------------------------
# METRICS
# -------------------------------------
st.subheader("📊 Overview Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", len(df))
c2.metric("Fraud Detected", df["fraud_flag"].sum())
c3.metric("High Risk", (df["risk_level"]=="High").sum())
c4.metric("Fraud Rate (%)", round(df["fraud_flag"].mean()*100,2))

# -------------------------------------
# RISK DISTRIBUTION
# -------------------------------------
st.subheader("📉 Risk Score Distribution")
fig, ax = plt.subplots()
ax.hist(df["risk_score"], bins=40)
st.pyplot(fig)

# -------------------------------------
# DEPARTMENT HEATMAP
# -------------------------------------
st.subheader("🏢 Department Risk Summary")
dept_risk = df.groupby("department_id")["risk_score"].mean()
st.bar_chart(dept_risk)

# -------------------------------------
# VENDOR WATCHLIST
# -------------------------------------
st.subheader("🚩 Vendor Watchlist")
vendor_watch = df[df["fraud_flag"]==1]["vendor_id"].value_counts().head(10)
if vendor_watch.empty:
    st.info("No risky vendors found")
else:
    st.bar_chart(vendor_watch)

# -------------------------------------
# TIME-BASED PATTERNS
# -------------------------------------
st.subheader("⏰ Fraud by Hour")
hour_fraud = df.groupby("hour")["fraud_flag"].mean()
st.line_chart(hour_fraud)

# -------------------------------------
# FLAGGED TRANSACTIONS
# -------------------------------------
st.subheader("📄 Flagged Transactions for Audit")

flagged = df[df["fraud_flag"]==1][[
    "transaction_id","department_id","vendor_id",
    "amount","risk_score","risk_level","explanation"
]]

if flagged.empty:
    st.info("No suspicious transactions")
else:
    st.dataframe(flagged)
