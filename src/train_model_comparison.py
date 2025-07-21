import joblib
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional models
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📌 {name} Evaluation:")
    print(f"🔹 MAE:  {mae:.2f}")
    print(f"🔹 RMSE: {rmse:.2f}")
    print(f"🔹 R²:   {r2:.4f}")
    
    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2}

def main():
    # Load processed data
    X_train = joblib.load("data/X_train.pkl")
    X_test = joblib.load("data/X_test.pkl")
    y_train = joblib.load("data/y_train.pkl")
    y_test = joblib.load("data/y_test.pkl")

    os.makedirs("models", exist_ok=True)

    # Define models to compare
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
    }

    if XGBRegressor:
        models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    if LGBMRegressor:
        models["LightGBM"] = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    results = []

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"\n✅ {name} training complete.")
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

        # Save each model
        model_path = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_path)
        print(f"💾 Saved {name} model to {model_path}")

    # Select best model based on RMSE
    best_model = min(results, key=lambda x: x["rmse"])
    joblib.dump(best_model["model"], "models/best_model.pkl")
    print(f"\n🏆 Best model: {best_model['name']} (saved as models/best_model.pkl)")

if __name__ == "__main__":
    main()
