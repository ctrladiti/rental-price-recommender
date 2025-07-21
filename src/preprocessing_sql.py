import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import re
import os

from db_connect import get_connection  # 🔁 DB connection here


def clean_house_size(value):
    if isinstance(value, str):
        match = re.search(r'\d+', value.replace(',', ''))
        if match:
            return float(match.group())
    return value


def clean_security_deposit(value):
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "no deposit":
            return 0.0
        value = re.sub(r"[^\d]", "", value)  # Remove ₹, commas, etc.
        if value:
            return float(value)
        return np.nan
    return value


def load_data_from_sql(city_filter=None):
    query = "SELECT * FROM rental_data"
    if city_filter:
        query += f" WHERE city = '{city_filter}'"

    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()

    # Clean & preprocess
    df.drop(columns=["city"], inplace=True, errors='ignore')
    df['house_size'] = df['house_size'].apply(clean_house_size)
    df['SecurityDeposit'] = df['SecurityDeposit'].apply(clean_security_deposit)

    df.dropna(subset=["price"], inplace=True)
    df.dropna(subset=["house_size", "SecurityDeposit", "numBathrooms"], inplace=True)

    return df


def preprocess_data(df):
    # Separate features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Identify columns
    numeric_features = ["house_size", "numBathrooms", "SecurityDeposit"]
    categorical_features = ["house_format", "house_type", "location", "Status"]

    # Build transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform
    X_processed = pipeline.fit_transform(X)

    # Save pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/preprocessing_pipeline.pkl")

    return X_processed, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = load_data_from_sql()  # Optional: add city_filter="Delhi"
    print(f"✅ Data loaded from SQL with {df.shape[0]} rows and {df.shape[1]} columns.")

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("✅ Data preprocessing complete.")
    print("🔹 X_train shape:", X_train.shape)
    print("🔹 X_test shape:", X_test.shape)
    print("📦 Preprocessing pipeline saved to 'models/preprocessing_pipeline.pkl'")
