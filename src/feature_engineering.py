import pandas as pd

def load_and_preprocess_data(path, drop_customer_id=True):
    # Load and basic clean
    df = pd.read_csv(path)
    df = df.dropna()
    df = df[df['TotalCharges'] != ' ']
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop irrelevant column
    if drop_customer_id:
        df = df.drop(columns=['customerID'])

    # Replace "No internet service" and "No phone service" with "No"
    replace_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in replace_cols:
        df[col] = df[col].replace({
            "No internet service": 'No',
            "No phone service": 'No'
        })

    # Binary encoding for Yes/No columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'] + replace_cols
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encoding for categorical features
    cat_cols = ['InternetService', 'Contract', 'PaymentMethod', 'gender']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
