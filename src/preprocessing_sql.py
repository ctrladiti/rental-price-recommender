import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from db_connect import load_data_from_sql

def main():
    # Load data from SQL
    df = load_data_from_sql()

    # Drop rows with missing target value
    df = df.dropna(subset=['price'])

    # Features and Target
    X = df.drop(columns=['id', 'price', 'Status'])
    y = df['price']

    # Separate column types
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Fit + transform
    X_processed = preprocessor.fit_transform(X)

    # Final check for NaNs
    if np.isnan(X_processed).any():
        print("⚠️ Warning: NaNs present even after imputation!")
        return

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Save processed data
    joblib.dump(X_train, "data/X_train.pkl")
    joblib.dump(X_test, "data/X_test.pkl")
    joblib.dump(y_train, "data/y_train.pkl")
    joblib.dump(y_test, "data/y_test.pkl")

    print("✅ Preprocessing complete and data saved.")

if __name__ == "__main__":
    main()
