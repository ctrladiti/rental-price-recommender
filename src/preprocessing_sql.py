import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from db_connect import load_data_from_sql
import os

def preprocess_data(df):
    df = df.copy()

    # Drop irrelevant columns
    columns_to_drop = ['currency', 'description', 'verificationDate', 'Status']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Handle "No Deposit" → 0 and cast
    df['SecurityDeposit'] = df['SecurityDeposit'].replace("No Deposit", 0)
    df['SecurityDeposit'] = pd.to_numeric(df['SecurityDeposit'], errors='coerce')

    # Drop rows with missing target
    df.dropna(subset=['price'], inplace=True)

    # Fill other missing values
    df.fillna({
        'house_type': 'Unknown',
        'house_size': 'Unknown',
        'numBathrooms': 1,
        'numBalconies': 0,
        'isNegotiable': 'No',
        'SecurityDeposit': df['SecurityDeposit'].median()
    }, inplace=True)

    # Define features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Categorical and numerical columns
    categorical = X.select_dtypes(include='object').columns.tolist()
    numerical = X.select_dtypes(include=np.number).columns.tolist()

    pipeline = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ])

    X_processed = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/preprocessing_pipeline.pkl")

    print("✅ Data preprocessing complete.")
    print(f"🔹 X_train shape: {X_train.shape}")
    print(f"🔹 X_test shape: {X_test.shape}")
    print("📦 Preprocessing pipeline saved to 'models/preprocessing_pipeline.pkl'")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data_from_sql()
    preprocess_data(df)
