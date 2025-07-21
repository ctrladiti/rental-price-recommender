import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate():
    # Load data
    X_test = joblib.load("data/X_test.pkl")
    y_test = joblib.load("data/y_test.pkl")

    # Load best model
    model = joblib.load("models/best_model.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Evaluation on Test Data:")
    print(f"🔹 MAE  (Mean Absolute Error): ₹{mae:.2f}")
    print(f"🔹 RMSE (Root Mean Squared Error): ₹{rmse:.2f}")
    print(f"🔹 R² Score (Accuracy-style metric): {r2:.4f}")

if __name__ == "__main__":
    evaluate()
