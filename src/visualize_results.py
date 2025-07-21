import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def visualize():
    # Load test data and model
    X_test = joblib.load("data/X_test.pkl")
    y_test = joblib.load("data/y_test.pkl")
    model = joblib.load("models/best_model.pkl")

    # Predict
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Setup plotting style
    sns.set(style="whitegrid")

    # 1. 📈 Actual vs. Predicted Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='navy', edgecolor='white')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    plt.xlabel("Actual Rent")
    plt.ylabel("Predicted Rent")
    plt.title("📈 Actual vs. Predicted Rent")
    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png")
    plt.show()

    # 2. 📉 Residuals Plot
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.title("📉 Distribution of Residuals")
    plt.tight_layout()
    plt.savefig("plots/residuals_distribution.png")
    plt.show()

    # 3. (Optional) Print Evaluation Summary
    print("\n📊 Evaluation Metrics:")
    print(f"🔹 MAE:  ₹{mean_absolute_error(y_test, y_pred):.2f}")
    print(f"🔹 RMSE: ₹{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"🔹 R²:   {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)
    visualize()
