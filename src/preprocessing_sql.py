import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import re
import os
from db_connect import load_table_as_dataframe


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


def load_data():
    df = load_table_as_dataframe("rental_data")
    print(f"✅ Raw data: {df.shape}")
    print(df.dtypes)

    print("\n🔍 Sample house_size values BEFORE cleaning:")
    print(df["house_size"].dropna().unique()[:10])
    print("\n🔍 Sample SecurityDeposit values BEFORE cleaning:")
    print(df["SecurityDeposit"].dropna().unique()[:10])

    # Only apply cleaning if the column is object/string type
    if df['house_size'].dtype == 'object':
        df['house_size'] = df['house_size'].apply(clean_house_size)
    else:
        print("ℹ️ Skipped cleaning 'house_size' as it's already numeric")

    if df['SecurityDeposit'].dtype == 'object':
        df['SecurityDeposit'] = df['SecurityDeposit'].apply(clean_security_deposit)
    else:
        print("ℹ️ Skipped cleaning 'SecurityDeposit' as it's already numeric")

    print("\n✅ After cleaning numeric fields")
    print("✅ house_size nulls:", df['house_size'].isnull().sum())
    print("✅ SecurityDeposit nulls:", df['SecurityDeposit'].isnull().sum())
    print("✅ numBathrooms nulls:", df['numBathrooms'].isnull().sum())

    print(f"➡️ After cleaning numeric fields: {df.shape}")
    df.dropna(subset=["price"], inplace=True)
    print(f"➡️ After dropping missing price: {df.shape}")
    df.dropna(subset=["house_size", "SecurityDeposit", "numBathrooms"], inplace=True)
    print(f"➡️ After dropping missing numeric features: {df.shape}")
    df.dropna(subset=["house_format", "house_type", "location", "city", "Status"], inplace=True)
    print(f"➡️ After dropping missing categorical features: {df.shape}")

    if df.empty:
        print("❌ All data dropped during cleaning. Check your raw data and cleaning rules.")
        exit()

    return df


def preprocess_data(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    numeric_features = ["house_size", "numBathrooms", "SecurityDeposit"]
    categorical_features = ["house_format", "house_type", "location", "city", "Status"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline([('preprocessor', preprocessor)])
    X_processed = pipeline.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/preprocessing_pipeline.pkl")

    return X_processed, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = load_data()
    print(f"✅ Cleaned data: {df.shape[0]} rows")

    X, y = preprocess_data(df)
    print(f"✅ Preprocessed data: {X.shape}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    os.makedirs("data", exist_ok=True)
    joblib.dump(X_train, "data/X_train.pkl")
    joblib.dump(X_test, "data/X_test.pkl")
    joblib.dump(y_train, "data/y_train.pkl")
    joblib.dump(y_test, "data/y_test.pkl")

    print("✅ Preprocessing complete.")
    print("🔹 X_train shape:", X_train.shape)
    print("🔹 X_test shape:", X_test.shape)
    print("📦 Saved all preprocessed data and pipeline.")
