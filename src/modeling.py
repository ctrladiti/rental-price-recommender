import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X_train = joblib.load("data/X_train.pkl")
    X_test = joblib.load("data/X_test.pkl")
    y_train = joblib.load("data/y_train.pkl")
    y_test = joblib.load("data/y_test.pkl")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("📊 Evaluation:")
    print("  MAE :", mean_absolute_error(y_test, y_pred))
    print("  RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("  R²  :", r2_score(y_test, y_pred))

    joblib.dump(model, "models/linear_model.pkl")
    print("✅ Model saved to models/linear_model.pkl")

if __name__ == "__main__":
    main()
