import joblib
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # Load processed data
    X_train = joblib.load("data/X_train.pkl")
    X_test = joblib.load("data/X_test.pkl")
    y_train = joblib.load("data/y_train.pkl")
    y_test = joblib.load("data/y_test.pkl")

    # Load preprocessing pipeline (optional, if needed later)
    pipeline = joblib.load("models/preprocessing_pipeline.pkl")

    # Initialize model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate model
    print("\n📊 Model Evaluation:")
    print(f"🔹 MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Use np.sqrt to avoid 'squared' error
    print(f"🔹 RMSE: {rmse:.2f}")
    print(f"🔹 R²:   {r2_score(y_test, y_pred):.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rental_price_model.pkl")
    print("\n💾 Model saved to 'models/rental_price_model.pkl'")

if __name__ == "__main__":
    main()
