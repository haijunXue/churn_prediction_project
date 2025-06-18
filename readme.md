## 📊 Churn Prediction with XGBoost & Streamlit Dashboard

### 🔍 Overview

This project predicts customer churn for a telecom company using machine learning. It includes end-to-end data preprocessing, training of an XGBoost model, and a web-based dashboard for business users built with Streamlit.
![image](https://github.com/user-attachments/assets/4075b9b5-9f0a-4024-839a-99438e74bfd3)

---

### 🎯 Goal

To identify customers at high risk of churn and highlight key risk factors, enabling targeted retention strategies.

---

### 🧠 Features

* Data preprocessing pipeline (`feature_engineering.py`)
* Churn probability prediction using **XGBoost**
* Risk level segmentation (Low / Medium / High)
* Extraction of key churn drivers (e.g., "Month-to-Month", "No OnlineSecurity")
* **Top 10 high-risk customers** export as CSV
* Interactive dashboard built with **Streamlit**

---

### 📁 Project Structure

```
churn_prediction_project/
├── data/
│   ├── telco_churn.csv
│   ├── top_10_high_risk.csv
│   └── risk_segmented_customer.csv
├── models/
│   └── xgboost_model.pkl
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── generate_predictions.py
│   └── dashboard_app.py
└── README.md
```

---

### ⚙️ How to Run

#### 1. Setup

```bash
pip install -r requirements.txt
```

#### 2. Train the model

```bash
python src/train_model.py
```

#### 3. Generate predictions

```bash
python src/generate_predictions.py
```

#### 4. Launch dashboard

```bash
streamlit run src/dashboard_app.py
```

---

### 📊 Dashboard Features

* **Churn Risk Segmentation**: Visualizes customer segments by risk level
* **Top 10 At-Risk Customers**: Table with churn probability and key risk factor
* **Churn Probability Distribution**: Histogram of predicted churn probabilities
* **Actionable Insights**: Identify patterns like contract type or security services that increase churn risk

---

### 🛠️ Tech Stack

* Python, Pandas, Scikit-learn, XGBoost
* Streamlit for web dashboard
* Feature Engineering Pipeline
* Joblib for model persistence

---

### 📈 Example Output

*(Include a screenshot here:)*

```
![Churn Dashboard Screenshot](path_or_link_to_screenshot.png)
```

---

### ✨ Future Work

* Add SHAP values for explainability
* Improve feature selection with domain knowledge
* Automate alerts for high-risk customers

---


