import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Path
project_root = Path(__file__).parent.parent
csv_path = project_root / 'data' / 'risk_segmented_customer.csv'
top_10_path = project_root / 'data' / 'top_10_high_risk.csv'

# Load data
df = pd.read_csv(csv_path)
top_10 = pd.read_csv(top_10_path)

# Calculate total revenue at risk
df["Revenue_Risk"] = df["MonthlyCharges"]* (df["Churn_Probability"] / 100)
total_revenue_risk = df["Revenue_Risk"].sum()
current_customers_at_risk = df[df["Risk_Level"] == "High"].shape[0]
most_valued_customer = df[df["Churn_Probability"] == df["Churn_Probability"].max()]["customerID"].values[0]

# Sidebar
st.sidebar.title("🔍 Filter")
risk_filter = st.sidebar.multiselect("Select Risk Level:", df["Risk_Level"].unique(), default=list(df["Risk_Level"].unique()))
filtered_df = df[df["Risk_Level"].isin(risk_filter)]

# Header
st.title("📉 Customer Churn Dashboard")

# Revenue risk & stats
col1, col2, col3 = st.columns(3)
col1.metric("💸 Current Revenue Risk", f"${total_revenue_risk:,.0f}")
col2.metric("👤 Customers at High Risk", current_customers_at_risk)
col3.metric("🌟 Top Customer at Risk", most_valued_customer)

# Churn by Spend
spend_bins = pd.cut(df["MonthlyCharges"], bins=[0, 50, 100, 150, 200], labels=["0-50", "50-100", "100-150", "150-200"])
spend_df = df.groupby(spend_bins, observed=False).agg({"Revenue_Risk": "sum", "customerID": "count"}).reset_index()

spend_df.rename(columns={"customerID": "Customers"}, inplace=True)
fig_spend = px.bar(spend_df, x="MonthlyCharges", y="Revenue_Risk", text="Customers",
                   title="Churn Analytics by Spend", labels={"Revenue_Risk": "Revenue at Risk"})
st.plotly_chart(fig_spend, use_container_width=True)

# Risk level pie chart
fig_pie = px.pie(df, names="Risk_Level", title="Risk Level Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# Top 10 customers at risk
st.subheader("🔝 Top 10 High-Risk Customers")
st.dataframe(top_10, use_container_width=True)

# Download buttons
st.download_button("📥 Download Full Risk Segmented Data", data=df.to_csv(index=False), file_name="risk_segmented_customer.csv")
st.download_button("📥 Download Top 10 High Risk Customers", data=top_10.to_csv(index=False), file_name="top_10_high_risk.csv")

# Optional: show data
with st.expander("🔍 Show Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)
