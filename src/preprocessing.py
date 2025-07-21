import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import re
import os


def clean_house_size(value):
    if isinstance(value, str):
        match = re.search(r'\d+', value.replace(',', ''))
        if match:
            return float(match.group())
    return np.nan


def clean_security_deposit(value):
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "no deposit":
            return 0.0
        value = re.sub(r"[^\d]", "", value)
        return float(value) if value else np.nan
    return value


def load_data(filepath):
    df = pd.read_csv(filepath)

    # Clean columns
    df['house_size'] = df['house_size'].apply(clean_house_size)
    df['SecurityDeposit'] = df['SecurityDeposit'].apply(clean_security_deposit)

    # Drop rows with missing target
    df.dropna(subset=["price"], inplace=True)

    # Drop rows where essential features are missing
    df.dropna(subset=["house_size", "SecurityDeposit", "numBathrooms"], inplace=True)

    # Drop rows with missing values in categorical columns
    df.dropna(subset=["house_format", "house_type", "location", "city", "Status"], inplace=True)

    return df


def preprocess_data(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    numeric_features = ["house_size", "numBathrooms", "SecurityDeposit"]
    categorical_features = ["house_format", "house_type", "location", "city", "Status"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_processed = pipeline.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/preprocessing_pipeline.pkl")

    return X_processed, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = load_data("data/Indian_housing_Delhi_data.csv")
    print(f"✅ Data loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    joblib.dump(X_train, "data/X_train.pkl")
    joblib.dump(X_test, "data/X_test.pkl")
    joblib.dump(y_train, "data/y_train.pkl")
    joblib.dump(y_test, "data/y_test.pkl")

    print("✅ Data preprocessing complete.")
    print("🔹 X_train shape:", X_train.shape)
    print("🔹 X_test shape:", X_test.shape)
    print("📦 Saved all preprocessed data and pipeline.")
