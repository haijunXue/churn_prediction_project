import pandas as pd
import joblib
from feature_engineering import load_and_preprocess_data
from pathlib import Path

project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'telco_churn.csv'
data_file = project_root / 'data'
models_path = project_root / 'models'

# Load model and data
model = joblib.load(models_path / 'xgboost_model.pkl')
df = load_and_preprocess_data(data_path, drop_customer_id=False)
customer_ids = pd.read_csv(data_path)["customerID"]
#X = df.drop("Churn", axis=1)
X = df.drop(columns=["Churn", "customerID"])


# Predict probabilities

probs = model.predict_proba(X)[:, 1]

# Categorize risk
risk = pd.cut(probs, bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])

# Reload raw data for reporting
df_raw_all= pd.read_csv(data_path)
df_raw = df_raw_all.loc[df.index].reset_index(drop=True)  # align with preprocessed rows
df_raw["Churn_Probability"] = (probs*100).round(2)
df_raw["Risk_Level"]= risk

# Simple Key Risk Factor extraction
df_raw["Key_Risk_Factor"] = df_raw.apply(lambda row:
                                         "No OnlineSecurity" if row["OnlineSecurity"]=="No" else
                                         "Month-to-Month" if row["Contract"] == "Month-to-month" else "N/A", axis=1)

# Save top 10 high-risk customers
top_10 = df_raw[df_raw["Risk_Level"] == "High"].sort_values("Churn_Probability", ascending=False)
top_10[["customerID", "Churn_Probability", "Key_Risk_Factor"]].to_csv(data_file / 'top_10_high_risk.csv', index=False)

# Save full risk-segmented data
df_raw.to_csv(data_file / 'risk_segmented_customer.csv', index=False)

print("✅ Prediction files created:")
print("- data/top_10_high_risk.csv")
print("- data/risk_segmented_customers.csv")