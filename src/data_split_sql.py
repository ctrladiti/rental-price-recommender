import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from db_connect import load_data_from_sql

def split_data():
    df = load_data_from_sql()

    print(f"✅ Loaded {df.shape[0]} rows")

    # Drop unnecessary columns
    df = df.drop(columns=["id"])  # id not needed

    # Drop rows with missing target (price) or essential features
    df = df.dropna(subset=["price", "numBathrooms"])

    # Separate target
    y = df["price"]
    X = df.drop(columns=["price"])

    # One-hot encode categorical columns
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("📦 Shapes:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)
    print("  y_train:", y_train.shape)
    print("  y_test :", y_test.shape)

    # Save splits
    os.makedirs("data", exist_ok=True)
    joblib.dump(X_train, "data/X_train.pkl")
    joblib.dump(X_test, "data/X_test.pkl")
    joblib.dump(y_train, "data/y_train.pkl")
    joblib.dump(y_test, "data/y_test.pkl")

    print("✅ Data saved in 'data/' folder.")

if __name__ == "__main__":
    split_data()
