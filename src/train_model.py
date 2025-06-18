import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from pathlib import Path
from feature_engineering import load_and_preprocess_data

project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'telco_churn.csv'
models_path = project_root / 'models'
def load_data():
    df = load_and_preprocess_data(data_path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n{name} Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))


def train_logistic_regression_model(X_train, X_test, y_train, y_test):
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    evaluate_model("Logistic Regression", model, X_test, y_test)
    joblib.dump(model, models_path / 'logistic_regression_model.pkl')

def train_random_forest_model(X_train, X_test, y_train, y_test):
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(pd.Series(model.feature_importances_, index=X_train.columns))
    evaluate_model("Random Forest", model, X_test, y_test)
    joblib.dump(model, models_path/ 'random_forest_model.pkl')

def train_xgboost_model(X_train, X_test, y_train, y_test):
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    evaluate_model("XGBoost", model, X_test, y_test)
    joblib.dump(model, models_path/ 'xgboost_model.pkl')

if __name__ == "__main__":
    models_path.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train all models
    train_logistic_regression_model(X_train, X_test, y_train, y_test )
    train_random_forest_model(X_train, X_test, y_train, y_test)
    train_xgboost_model(X_train, X_test, y_train, y_test)
    print("\n✅ All models trained and saved successfully.")