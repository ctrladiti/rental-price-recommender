# 🏙️ Rental Price Recommendation System for Urban Areas

A data-driven machine learning solution to predict residential rental prices in major Indian cities (Delhi, Mumbai, Pune) based on area, square footage, house type, amenities, and more.

---

## 📌 Overview

With rising urban housing demands, estimating fair rental prices is challenging. This project analyzes rental housing datasets and predicts accurate rent prices using regression models trained on location, house features, and amenities.

---

## 🎯 Key Features

- 📊 Exploratory Data Analysis from MySQL database
- 🧹 Data preprocessing with pipelines (encoding, imputation, scaling)
- 🤖 Model comparison (Linear Regression, Random Forest, Gradient Boosting)
- ✅ Selected best model using RMSE and R² metrics
- 🧠 Rent prediction on new user input or batch CSV
- 📈 Visualization: Actual vs Predicted + Residuals
- 🖥️ (Optional) Streamlit-based prediction app

---

## 🛠️ Tech Stack

| Tool                       | Purpose                       |
| -------------------------- | ----------------------------- |
| **Python**                 | Core programming language     |
| **MySQL**                  | Backend data storage          |
| **Pandas, NumPy**          | Data handling                 |
| **Scikit-learn**           | ML modeling and evaluation    |
| **Matplotlib, Seaborn**    | Visualizations                |
| **Joblib**                 | Model/pipeline persistence    |
| **Streamlit** _(optional)_ | Web interface for predictions |

---

## 🗂️ Project Structure

```

rental-price-recommender/
│
├── data/
│ ├── X_train.pkl, X_test.pkl
│ ├── y_train.pkl, y_test.pkl
│
├── models/
│ ├── preprocessing_pipeline.pkl
│ ├── best_model.pkl
│
├── plots/
│ ├── actual_vs_predicted.png
│ ├── residuals_distribution.png
│
├── src/
│ ├── db_connect.py # SQL connection helper
│ ├── eda_sql.py # Data exploration from MySQL
│ ├── preprocessing_sql.py # Data cleaning & transformation
│ ├── train_model_comparison.py # Model training & selection
│ ├── evaluate_model.py # Final evaluation
│ ├── predict.py # Single prediction interface
│ ├── visualize_results.py # Diagnostic plots
│ ├── app.py # (Optional) Streamlit UI

```

---

## 🚀 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Aditi-1304/rental-price-recommender.git
cd rental-price-recommender

# 2. Create virtual environment & activate
python -m venv venv
venv\Scripts\activate   # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training and evaluation
python src/train_model_comparison.py
python src/evaluate_model.py

# 5. Predict using sample input
python src/predict.py

# 6. (Optional) Launch Streamlit App
streamlit run src/app.py
```

---

## 📈 Results

| Model             | RMSE (₹) | R² Score |
| ----------------- | -------- | -------- |
| Linear Regression | 68,200+  | 0.81     |
| Random Forest     | 56,530   | 0.91 ✅  |
| Gradient Boosting | 58,000   | 0.89     |

📊 **Best Model:** Random Forest
📉 **Average Prediction Error (MAE):** ₹20,719.63

![Actual vs Predicted](plots/actual_vs_predicted.png)
![Residuals](plots/residuals_distribution.png)

---

## 🧪 Sample Prediction

```bash
🏠 Area: Kothrud, Pune
📐 Size: 1200 sq ft, 2 BHK
🏗️ Type: Apartment, Semi-Furnished
📦 Amenities: Gym, Parking

➡️ Predicted Rent: ₹38,200/month
```

---

## 📚 Future Enhancements

- Integrate geospatial features (distance to city center, metro stations)
- Deploy with CI/CD pipeline
- Host app on Streamlit Cloud or Hugging Face Spaces
- Add Explainable AI (SHAP) for feature transparency

---

## 🙋‍♀️ About Me

👩‍💻 **Aditi Agrawal**
Final-year Engineering student | Data Science + Web Dev Enthusiast <br>
🔗 [GitHub](https://github.com/Aditi-1304) | [LeetCode](https://leetcode.com/u/Aditi_786/) | [HackerRank](https://www.hackerrank.com/profile/aditi786aaa)
