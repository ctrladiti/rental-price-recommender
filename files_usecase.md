# Usecases of Different Files of the project

---

## 🔹 `data_loader.py` – Rental Price Recommendation System

### ▶ Purpose

The `data_loader.py` file is responsible for **loading rental housing data from a CSV file** and **inserting it into the MySQL `rental_data` table**. This allows your system to maintain a centralized, clean dataset for EDA, preprocessing, and modeling.

---

### ▶ Function: `insert_data(filepath, city)`

**Parameters:**

- `filepath`: Path to the CSV file containing rental property data.
- `city`: Name of the city (added as a column if not present in CSV).

---

### ▶ Step-by-Step What It Does:

1. **Reads CSV File:**

   ```python
   df = pd.read_csv(filepath)
   ```

   Loads data into a pandas DataFrame from the provided file path.

2. **Cleans Column Names:**

   ```python
   df.columns = df.columns.str.strip()
   ```
   
   Removes extra spaces from column headers to prevent key errors.

3. **Adds `city` Column (if missing):**

   ```python
   if 'city' not in df.columns:
       df['city'] = city
   ```

   Ensures that the city name is included in every row.

4. **Handles Missing Values:**

   ```python
   df = df.replace({pd.NA: None, 'nan': None, 'NaN': None, '': None})
   df = df.where(pd.notnull(df), None)
   ```

   Replaces missing values with `None` to ensure compatibility with SQL insertion.

5. **Database Connection:**

   ```python
   conn = get_connection()
   cursor = conn.cursor()
   ```

   Establishes a connection using `get_connection()` from your `db_connect.py`.

6. **Inserts Rows into MySQL Table:**

   - Iterates through each row of the DataFrame.
   - Converts row to dictionary and prepares SQL-safe values.
   - Executes an `INSERT INTO rental_data (...) VALUES (...)` SQL query.

7. **Closes Connection:**

   ```python
   conn.commit()
   cursor.close()
   conn.close()
   ```

   Finalizes the transaction and safely closes the database connection.

---

### ▶ Columns in the CSV and used in DB insert:

- `house_format`
- `house_type`
- `house_size`
- `location`
- `city`
- `price`
- `numBathrooms`
- `SecurityDeposit`
- `Status`

---

### ▶ Summary

The `data_loader.py` script is a crucial **data ingestion utility** for the project. It ensures raw rental data is cleaned, validated, and inserted into the MySQL `rental_data` table — forming the foundational dataset for all downstream analytics and ML workflows.

---
---

## 🔹 `main.py` – Rental Price Data Loader Runner

### ▶ Purpose

The `main.py` script serves as the **entry point** to **load and insert rental housing data for multiple cities** into the MySQL `rental_data` table using the `insert_data()` function from the `data_loader.py` file.

---

### ▶ Functionality

###  It does the following:

1. **Imports pandas** for general use (though not directly used here).

   ```python
   import pandas as pd
   ```

2. **Imports `insert_data()`** from your custom data loader:

   ```python
   from src.data_loader import insert_data
   ```

3. **Defines a `main()` function** that:

   - Loads and inserts Delhi rental data.
   - Loads and inserts Mumbai rental data.
   - Loads and inserts Pune rental data.
     Each call uses the corresponding city-specific CSV file and injects the city name where needed.

   ```python
   def main():
       insert_data("data/Indian_housing_Delhi_data.csv", "Delhi")
       insert_data("data/Indian_housing_Mumbai_data.csv", "Mumbai")
       insert_data("data/Indian_housing_Pune_data.csv", "Pune")
   ```

4. **Executes `main()` only when run as a script:**

   ```python
   if __name__ == "__main__":
       main()
   ```

---

### ▶ How to Run

```bash
python main.py
```

> Make sure your Python environment is active and your MySQL database is running with proper credentials in `db_connect.py`.

---

### ▶ Summary

- The script helps automate the **loading and insertion of city-specific rental data** into your SQL database.
- You can run it **once to populate the `rental_data` table** with cleaned data from Delhi, Mumbai, and Pune.
- It relies on `src.data_loader.insert_data()` to handle the actual insertion logic.

---
---

## 🔹 `eda_sql.py` – Exploratory Data Analysis (EDA) from SQL

### ▶ Purpose

The `eda_sql.py` script is responsible for performing **Exploratory Data Analysis (EDA)** on the **rental data loaded directly from your SQL database** using the `load_data_from_sql()` function. It generates summary statistics, missing value reports, and visualizations, which are saved locally.

---

### ▶ Dependencies Used

- `pandas`: For data manipulation.
- `seaborn` and `matplotlib`: For plotting.
- `os`: To create output directories.
- `db_connect.load_data_from_sql`: Loads data from your MySQL table `rental_data`.

---

### ▶ Function: `perform_eda()` (What It Does)

1. ### **Loads Data**

   ```python
   df = load_data_from_sql()
   ```

   Fetches the rental data from the database into a pandas DataFrame.

2. ### **Basic Data Overview**

   - Shows number of rows and columns.
   - Displays `.info()`, `.head()`, `.isnull()`, `.describe()`, and unique categorical values.

3. ### **Missing Values**

   - Prints both **raw count** and **percentage of missing data**.

4. ### **Creates Output Folder**

   ```python
   os.makedirs("eda_outputs", exist_ok=True)
   ```

   Ensures that a folder `eda_outputs/` exists to store the plots.

5. ### **Visualizations Generated:**

   - **Histograms** for numeric columns (`price`, `numBathrooms`, etc.).
   - **Countplots** for top 10 values in each categorical column (e.g., `city`, `house_type`).
   - **Boxplot** of `price` by `Status` (e.g., Furnished/Unfurnished).
   - **Correlation Heatmap** for numeric variables.
   - **Barplot** showing average price in **top 10 locations** by listing count.

6. ### **Saves All Plots**

   - Visualizations are saved inside `eda_outputs/` folder as `.png` files.
   - Example: `price_distribution.png`, `Status_countplot.png`, `correlation_heatmap.png`, etc.

---

### ▶ Output Folder

All visualizations are saved in:

```
eda_outputs/
├── price_distribution.png
├── Status_countplot.png
├── price_by_status.png
├── correlation_heatmap.png
├── avg_price_by_top_locations.png
...
```

---

### ▶ How to Run

From terminal:

```bash
python eda_sql.py
```

> Make sure your MySQL DB is running and the `load_data_from_sql()` function is correctly configured to pull data from the `rental_data` table.

---

### ▶ Summary

This script provides a **quick and automated way to understand your SQL rental dataset**, identify potential issues like missing values, and generate useful visualizations that inform preprocessing and modeling.

---
---

## 🔹 `preprocessing.py` – Data Cleaning and Preprocessing Pipeline

### ▶ Purpose

This script performs the **full preprocessing pipeline** for your rental data project:

- Cleans raw data
- Handles missing values
- Transforms features (scaling + encoding)
- Splits data into train/test sets
- Saves outputs for modeling

---

### ▶ Key Functions

■ `clean_house_size(value)`

Cleans `house_size` by extracting the numeric part from strings like `"1,200 sqft"` → `1200`.

---

■ `clean_security_deposit(value)`

Converts deposit strings like:

- `"No Deposit"` → `0.0`
- `"₹10,000"` → `10000.0`

---

■ `load_data(filepath)`

1. Loads a CSV file.
2. Applies cleaning functions.
3. Drops rows missing critical features (`price`, `house_size`, `SecurityDeposit`, etc.).

- Returns a **cleaned DataFrame**.

---

■ `preprocess_data(df)`

1. Separates features (`X`) and target (`y = price`).
2. Identifies:

   - Numeric: `house_size`, `numBathrooms`, `SecurityDeposit`
   - Categorical: `house_format`, `house_type`, `location`, `city`, `Status`

3. Applies:

   - `StandardScaler` on numeric
   - `OneHotEncoder` on categorical

4. Creates and fits a **ColumnTransformer-based pipeline**.
5. Saves the pipeline as `models/preprocessing_pipeline.pkl`.

- Returns: Transformed features `X_processed`, target `y`

---

■ `split_data(X, y, test_size=0.2)`

Splits `X` and `y` into:

- `X_train`, `X_test`
- `y_train`, `y_test`

Using an 80-20 split (default), with reproducibility via `random_state=42`.

---

### ▶ When Run as a Script

```bash
python preprocessing.py
```

It performs the following:

1. Loads and cleans data from `data/Indian_housing_Delhi_data.csv`.
2. Applies preprocessing (scaling + encoding).
3. Splits into train/test.
4. Saves the following as `.pkl` files:

   - `data/X_train.pkl`
   - `data/X_test.pkl`
   - `data/y_train.pkl`
   - `data/y_test.pkl`
   - `models/preprocessing_pipeline.pkl`

---

### ▶ Summary

| Stage         | Output                                  |
| ------------- | --------------------------------------- |
| Cleaned Data  | pandas DataFrame                        |
| Preprocessing | Transformed features (Num + Cat)        |
| Pipeline      | Saved as `preprocessing_pipeline.pkl`   |
| Train/Test    | 4 `.pkl` files for training and testing |

This file prepares your data for modeling and is essential to maintain consistency in input transformations at both training and prediction time.

---

# 📄 `preprocessing_sql.py` – Preprocessing Pipeline Using SQL Data

## 🔍 Purpose

This script is designed to **load rental housing data directly from a SQL database**, clean and preprocess it, and save the processed data and pipeline for later modeling.

---

## 🧩 Key Functionalities

### ✅ 1. `load_data()`

- **Reads from MySQL** using:

  ```python
  from db_connect import load_table_as_dataframe
  ```

  Retrieves data from the `rental_data` table.

- **Cleans key numeric fields:**

  - `house_size`: Extracts numeric part (e.g., `"1,200 sqft"` → `1200`).
  - `SecurityDeposit`: Converts values like `"No Deposit"` or `"₹15,000"` to numeric.

- **Drops rows with missing or invalid values** in critical features:

  - Target (`price`)
  - Numeric: `house_size`, `SecurityDeposit`, `numBathrooms`
  - Categorical: `house_format`, `house_type`, `location`, `city`, `Status`

- **Prints intermediate shapes and issues** to help debug if too much data is dropped.

> ❗ Exits if the cleaned dataset ends up empty.

---

### ✅ 2. `preprocess_data(df)`

- Splits data into:

  - Features (`X`)
  - Target (`y = price`)

- Defines feature types:

  - **Numeric**: `house_size`, `numBathrooms`, `SecurityDeposit`
  - **Categorical**: `house_format`, `house_type`, `location`, `city`, `Status`

- Creates a **ColumnTransformer pipeline**:

  - `StandardScaler` for numeric features
  - `OneHotEncoder` (non-sparse) for categorical features

- Fits the pipeline and transforms `X`

- Saves the full pipeline as:

  ```
  models/preprocessing_pipeline.pkl
  ```

---

### ✅ 3. `split_data(X, y)`

- Performs a train/test split (default: 80/20)
- Uses `random_state=42` for reproducibility

---

### ✅ 4. Script Execution (`__main__`)

When you run:

```bash
python preprocessing_sql.py
```

It will:

1. Load and clean data from SQL
2. Preprocess and transform it
3. Split into train/test
4. Save:

   - `data/X_train.pkl`
   - `data/X_test.pkl`
   - `data/y_train.pkl`
   - `data/y_test.pkl`
   - `models/preprocessing_pipeline.pkl`

---

## 🧠 Summary

| Stage        | Output                                   |
| ------------ | ---------------------------------------- |
| Raw SQL      | `load_table_as_dataframe("rental_data")` |
| Cleaning     | Handles `house_size`, `SecurityDeposit`  |
| Transforming | ColumnTransformer (scaling + encoding)   |
| Splitting    | Train/test sets (`.pkl` files)           |
| Persisting   | Saves preprocessor and data for modeling |

This script ensures your ML pipeline uses **clean and consistently preprocessed data from your live database**, enabling more accurate and scalable predictions.

---

Here’s a **Markdown-formatted explanation** of your `db_connect.py` file and its role in your **Rental Price Recommendation System** project:

---

# 📄 `db_connect.py` – MySQL Database Connection Utility

## 🔍 Purpose

This file provides **database access functionality** by:

1. Establishing a connection to your MySQL database.
2. Loading entire tables or running specific SQL queries as pandas DataFrames.

---

## 🔧 MySQL Connection Setup

### ✅ `get_connection()`

```python
def get_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='rental_data'
    )
```

- Connects to the MySQL database `rental_data`.
- Uses local credentials: `user=root`, `password=root`.
- This function is **reused by all SQL access methods** in the project.

> 🔒 Consider using environment variables or a config file for credentials in production.

---

## 📥 Data Load Functions

### ✅ `load_data_from_sql()`

```python
def load_data_from_sql():
    ...
```

- Executes:

  ```sql
  SELECT * FROM rental_data;
  ```

- Returns the full table as a **pandas DataFrame**.
- Used for full-table access, especially in EDA and preprocessing.

---

### ✅ `load_table_as_dataframe(table_name: str)`

```python
def load_table_as_dataframe(table_name: str) -> pd.DataFrame:
    ...
```

- Dynamically queries any table name:

  ```sql
  SELECT * FROM <table_name>;
  ```

- Logs the query and the number of rows/columns returned.
- Useful for general-purpose, table-agnostic data loading.

---

## 🧠 Summary

| Function                              | Purpose                                  |
| ------------------------------------- | ---------------------------------------- |
| `get_connection()`                    | Returns a live connection to MySQL       |
| `load_data_from_sql()`                | Loads all rows from `rental_data` table  |
| `load_table_as_dataframe(table_name)` | Loads any given SQL table as a DataFrame |

This file is essential for all your **SQL → Python workflows**, powering EDA, preprocessing, and model integration.

---

Here's a **Markdown-formatted explanation** of your `data_split_sql.py` file and its role in your **Rental Price Recommendation System** project:

---

# 📄 `data_split_sql.py` – Direct SQL-Based Data Splitter

## 🔍 Purpose

This script **loads rental data from your MySQL database**, prepares it for training by handling missing values and encoding, then **splits the data into training and test sets** for model development.

---

## 🧩 What It Does

### ✅ 1. Load Data

```python
df = load_data_from_sql()
```

- Pulls the entire `rental_data` table using the helper function from `db_connect.py`.

---

### ✅ 2. Drop Unnecessary or Incomplete Rows

```python
df = df.drop(columns=["id"])
df = df.dropna(subset=["price", "numBathrooms"])
```

- Removes the `id` column (assumed to be an auto-increment primary key, not useful for prediction).
- Drops rows missing the **target column** (`price`) or an essential feature (`numBathrooms`).

---

### ✅ 3. Split Features and Target

```python
y = df["price"]
X = df.drop(columns=["price"])
```

- `X` is the input features.
- `y` is the rental price (target).

---

### ✅ 4. One-Hot Encode Categorical Variables

```python
categorical_cols = X.select_dtypes(include="object").columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
```

- Automatically detects all object-type columns.
- Applies one-hot encoding to convert them into numeric format.
- Uses `drop_first=True` to avoid dummy variable trap.

---

### ✅ 5. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```

- Splits data into 80% training, 20% testing sets.
- Uses a fixed `random_state=42` for reproducibility.

---

### ✅ 6. Save Output Files

```python
joblib.dump(..., "data/...")
```

- Saves all 4 components (`X_train`, `X_test`, `y_train`, `y_test`) into the `data/` folder as `.pkl` files.

---

## 📂 Output Files

| File Name          | Description                |
| ------------------ | -------------------------- |
| `data/X_train.pkl` | Features for training      |
| `data/X_test.pkl`  | Features for testing       |
| `data/y_train.pkl` | Target values for training |
| `data/y_test.pkl`  | Target values for testing  |

---

## ▶️ How to Run

```bash
python data_split_sql.py
```

> Make sure the MySQL database is running and accessible, and the `rental_data` table is populated.

---

## 🧠 Summary

| Task          | Description                                      |
| ------------- | ------------------------------------------------ |
| Data Source   | SQL table `rental_data`                          |
| Preprocessing | Drop `id`, handle missing values, one-hot encode |
| Splitting     | 80/20 train-test split                           |
| Output        | `.pkl` files saved in `data/`                    |

This script is a **minimal, SQL-driven alternative** to the full preprocessing pipeline — suitable when preprocessing is lightweight or already handled elsewhere.

---

Let me know if you'd like to:

- Include scaling
- Integrate this with a model training script
- Convert this into a modular function for reuse

---

Here's a **Markdown-formatted explanation** of your `modeling.py` file, which builds and evaluates your rental price prediction model:

---

# 🤖 `modeling.py` – Model Training and Evaluation Script

## 📌 Goal

Train a **Linear Regression model** on your preprocessed rental dataset and evaluate its performance.

---

## 🧩 What It Does

### ✅ 1. Load Preprocessed Data

```python
X_train = joblib.load("data/X_train.pkl")
X_test = joblib.load("data/X_test.pkl")
y_train = joblib.load("data/y_train.pkl")
y_test = joblib.load("data/y_test.pkl")
```

- Loads the data split outputs from `data_split_sql.py`.
- These are **ready-to-use** feature matrices and target vectors.

---

### ✅ 2. Train the Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

- A simple **Linear Regression** model from `sklearn` is used.
- Fits on the training data.

---

### ✅ 3. Make Predictions

```python
y_pred = model.predict(X_test)
```

- Predicts rental prices on the test set.

---

### ✅ 4. Evaluate Performance

```python
print("  MAE :", mean_absolute_error(y_test, y_pred))
print("  RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("  R²  :", r2_score(y_test, y_pred))
```

- **MAE (Mean Absolute Error)** – average absolute difference between predicted and actual prices.
- **RMSE (Root Mean Squared Error)** – penalizes larger errors.
- **R² (R-squared Score)** – indicates how well model explains the variation in target.

---

### ✅ 5. Save the Trained Model

```python
joblib.dump(model, "models/linear_model.pkl")
```

- Stores the trained model for later use (e.g., in prediction scripts or Streamlit app).

---

## 🗃️ Output

| File Path                 | Description                     |
| ------------------------- | ------------------------------- |
| `models/linear_model.pkl` | Trained Linear Regression model |

---

## ▶️ How to Run

```bash
python modeling.py
```

> Make sure `data/X_train.pkl`, `X_test.pkl`, etc., already exist and the `models/` folder is present or created by the script.

---

## 🧠 Summary

| Step        | Description                                   |
| ----------- | --------------------------------------------- |
| Load Data   | Pre-saved `X_train`, `X_test`, etc. from disk |
| Train Model | Scikit-learn Linear Regression                |
| Evaluate    | MAE, RMSE, R²                                 |
| Save Model  | Serialized to `.pkl` format in `models/`      |

This script forms the **core modeling step** of your project pipeline. You can later use this trained model for predictions in your Streamlit UI.

---

Let me know if you’d like to:

- Switch to another regression algorithm (e.g., `RandomForestRegressor`)
- Add cross-validation
- Log metrics to a file or dashboard

---

Here's a clean breakdown of your `train_model_comparision.py` script — it's a solid, extensible model benchmarking pipeline for your rental price prediction project.

---

# 🧠 `train_model_comparision.py` — Model Benchmarking & Selection

## 🎯 Objective

Train **multiple regression models**, evaluate their performance, and save:

- Each individual model.
- The **best performing model** (based on RMSE).

---

## 📦 Key Features

### ✅ Model Options Included

| Model               | Library      |
| ------------------- | ------------ |
| Linear Regression   | scikit-learn |
| Random Forest       | scikit-learn |
| Gradient Boosting   | scikit-learn |
| K-Nearest Neighbors | scikit-learn |
| XGBoost (optional)  | xgboost      |
| LightGBM (optional) | lightgbm     |

> ⚠️ `XGBoost` and `LightGBM` are imported conditionally. If not installed, they’re simply skipped.

---

## 🔍 Evaluation Metrics

Each model is evaluated using:

- `MAE` (Mean Absolute Error)
- `RMSE` (Root Mean Squared Error)
- `R²` (R-squared Score)

### 📊 Output Format Example

```plaintext
📌 Random Forest Evaluation:
🔹 MAE:  1032.52
🔹 RMSE: 1604.17
🔹 R²:   0.8653
```

---

## 🧪 Model Evaluation Logic

```python
def evaluate_model(name, model, X_test, y_test):
```

- Encapsulates metric calculation and reporting.
- Returns a result dictionary used for model selection.

---

## 🔄 Workflow Summary

### 1. Load Preprocessed Data

```python
X_train = joblib.load("data/X_train.pkl")
...
```

### 2. Train and Evaluate All Models

```python
for name, model in models.items():
    model.fit(X_train, y_train)
    ...
```

### 3. Save Each Model

```python
joblib.dump(model, model_path)
```

### 4. Select Best Model by Lowest RMSE

```python
best_model = min(results, key=lambda x: x["rmse"])
```

---

## 🗃️ Output Files

| File Path                              | Description                           |
| -------------------------------------- | ------------------------------------- |
| `models/linear_regression_model.pkl`   | Trained Linear Regression model       |
| `models/random_forest_model.pkl`       | Trained Random Forest model           |
| `models/gradient_boosting_model.pkl`   | Trained Gradient Boosting model       |
| `models/knn_regressor_model.pkl`       | Trained KNN model                     |
| `models/xgboost_model.pkl` (optional)  | Trained XGBoost model (if installed)  |
| `models/lightgbm_model.pkl` (optional) | Trained LightGBM model (if installed) |
| `models/best_model.pkl`                | 🚀 Best overall model (by RMSE)       |

---

## ▶️ How to Run

```bash
python train_model_comparision.py
```

> ✅ Make sure `data/X_train.pkl`, `X_test.pkl`, etc., are pre-generated.
> ✅ Create the `models/` directory manually **OR** it will be created by `os.makedirs()`.

---

## 💡 Suggestions

- ✅ **Add Logging to CSV/Excel**: Save `results` to a file for future reference.
- ✅ **Plot RMSE Bar Chart** using matplotlib or seaborn.
- 🚀 Optionally add **cross-validation** to make comparisons more robust.

Let me know if you’d like code for:

- Cross-validation based selection
- Saving result comparison as CSV
- Visualizing metrics per model in a bar chart

---

**TL;DR**: This script benchmarks multiple regression models, logs key metrics, and saves both individual models and the best performer — making it ideal for production-ready model selection.

---

Here is a Markdown (`.md`) formatted explanation for your `evaluate_model.py` file and how it fits into your **Rental Price Recommendation System** project:

---

## 📄 `evaluate_model.py` — Final Evaluation of Best Model

### 🔍 Purpose:

This script **evaluates the performance** of the **best trained regression model** on the test data using common regression metrics.

---

### 🛠️ What it Does:

1. **Loads test data**:

   - `X_test.pkl` and `y_test.pkl` from the `data/` directory are loaded using `joblib`.

2. **Loads the best model**:

   - It uses the trained model saved earlier as `models/best_model.pkl` (produced by `train_model_comparision.py`).

3. **Predicts rental prices**:

   - Makes predictions on the test set using the best model.

4. **Calculates evaluation metrics**:

   - **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual prices.
   - **RMSE** (Root Mean Squared Error): Measures error magnitude and penalizes larger errors.
   - **R² Score**: Indicates how well the model explains the variance in rental prices.

5. **Prints results**:

   - Shows a user-friendly summary of model performance with currency symbols (₹) for clarity.

---

### 🧩 How it Fits in the Project:

| File                         | Role                                                                               |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `db_connect.py`              | Connects to MySQL and loads the `rental_data` table as a DataFrame.                |
| `data_split_sql.py`          | Loads data from SQL, cleans, encodes, splits into train/test, and saves as `.pkl`. |
| `modeling.py`                | Trains a simple Linear Regression model.                                           |
| `train_model_comparision.py` | Trains multiple models, compares them, and saves the best one.                     |
| ✅ `evaluate_model.py`       | **Loads the best model and evaluates it on the test set.**                         |

---

### ✅ When to Use:

Run this file **after** training is complete (i.e., after `train_model_comparision.py` has saved `best_model.pkl`) to see how well the best model performs on unseen data.

---

### 💡 Example Output:

```
📊 Model Evaluation on Test Data:
🔹 MAE  (Mean Absolute Error): ₹1375.20
🔹 RMSE (Root Mean Squared Error): ₹1789.45
🔹 R² Score (Accuracy-style metric): 0.8427
```

---

Let me know if you want a similar `.md` explanation for other files too.

---

Here's a `.md` explanation for your `visualize_results.py` file in the context of your **Rental Price Recommendation System** project:

---

## 📄 `visualize_results.py` — Visual Evaluation of Model Performance

### 🎯 Purpose:

This script **generates visual plots** to assess how well the model performs in predicting rental prices by comparing predicted vs actual values and analyzing residuals.

---

### 🛠️ What it Does:

1. **Loads data and model**:

   - Loads `X_test.pkl`, `y_test.pkl` from `data/`
   - Loads the trained best model (`best_model.pkl`) from `models/`

2. **Predicts rental prices**:

   - Uses the model to predict on `X_test`
   - Calculates **residuals** (i.e., actual - predicted)

3. **Creates 2 diagnostic plots**:

   - **📈 Actual vs Predicted Plot**:

     - Scatter plot showing how closely the predictions match the actual rental values.
     - A red dashed reference line (`y = x`) helps visualize prediction accuracy.
     - Saved to: `plots/actual_vs_predicted.png`

   - **📉 Residuals Distribution Plot**:

     - Histogram (with KDE) of residuals to show error distribution.
     - Helps check for bias in predictions.
     - Saved to: `plots/residuals_distribution.png`

4. **Prints evaluation metrics**:

   - MAE, RMSE, and R² are also displayed for reference.

5. **Ensures plot folder exists**:

   - Automatically creates a `plots/` directory if it doesn’t already exist.

---

### 📊 Sample Output:

#### Console Output:

```
📊 Evaluation Metrics:
🔹 MAE:  ₹1358.72
🔹 RMSE: ₹1720.15
🔹 R²:   0.8539
```

#### Saved Files:

- `plots/actual_vs_predicted.png`
- `plots/residuals_distribution.png`

---

### 🧩 How it Fits in the Project:

| File                         | Role                                                 |
| ---------------------------- | ---------------------------------------------------- |
| `db_connect.py`              | Loads data from MySQL into DataFrames                |
| `data_split_sql.py`          | Prepares and saves train-test split data             |
| `modeling.py`                | Trains a simple model                                |
| `train_model_comparision.py` | Trains and selects the best model                    |
| `evaluate_model.py`          | Prints performance scores of best model              |
| ✅ `visualize_results.py`    | **Visualizes model predictions and residual errors** |

---

### ✅ When to Use:

Run this file after the best model is trained and saved. It provides visual confidence in the model's performance and helps detect issues like bias or poor fit.

---

Let me know if you'd like these markdown files saved into your repo or automatically generated summaries for other components.

---

Here’s a detailed `.md` explanation for your `app.py` file for the **Rental Price Recommendation System** project:

---

## 🌐 `app.py` — Streamlit UI for Rental Price Prediction

### 🎯 Purpose:

This script provides a **simple and interactive web interface** using Streamlit to allow users to input property details and get **instant rental price predictions** powered by your trained machine learning model.

---

### 💻 How It Works:

1. **🔃 Loads Pretrained Assets**

   - Loads `preprocessing_pipeline.pkl` to transform the input features.
   - Loads `best_model.pkl` — the ML model that makes predictions.

2. **🧾 Streamlit App Setup**

   - Configures page title and layout.
   - Displays a form for user input (property details).

3. **📥 Input Fields**

   - `house_format`: e.g., _2 BHK_
   - `house_type`: e.g., _Apartment_, _Villa_
   - `house_size`: numeric input like _1100 sqft_
   - `location`: text input for area or colony
   - `city`: choose from _Delhi_, _Mumbai_, or _Pune_
   - `numBathrooms`: numeric step input
   - `SecurityDeposit`: e.g., ₹25,000
   - `Status`: _Furnished_, _Semi-Furnished_, _Unfurnished_

4. **🧹 Input Cleaning**

   - Uses `clean_numeric()` to extract numeric values from strings (e.g., ₹ symbols, commas, “sqft”).
   - Ensures that values like `house_size` and `security_deposit` are clean floats.

5. **📈 Prediction**

   - All fields are converted to a single-row `DataFrame`.
   - Passed through the preprocessing pipeline.
   - Final rental price is predicted and shown to the user with proper formatting (`₹xx,xxx`).

6. **⚠️ Error Handling**

   - Catches exceptions and displays errors using `st.error()`.

---

### 📊 Output Example

When the user enters property details and clicks "Predict Rent":

```
💰 Estimated Monthly Rent: ₹28,500
```

Or, in case of error:

```
❌ Error during prediction: ValueError: ...
```

---

### 📁 Folder Dependency Checklist

| File/Folder                         | Description                                                    |
| ----------------------------------- | -------------------------------------------------------------- |
| `models/best_model.pkl`             | Trained model                                                  |
| `models/preprocessing_pipeline.pkl` | Pipeline with encoders, scalers, etc.                          |
| `app.py`                            | ✅ Streamlit UI                                                |
| `requirements.txt`                  | Should include `streamlit`, `joblib`, `pandas`, `scikit-learn` |

---

### 🚀 To Run the App

```bash
streamlit run app.py
```

> Make sure you're in the project root directory and the `models/` folder exists.

---

### ✅ When to Use

- Use this app **after training is complete** and you've saved your model and pipeline.
- It is ideal for **demoing the system**, collecting feedback, or deploying as a **proof of concept**.

---

Let me know if you want a `requirements.txt` or deployment guide (e.g., Streamlit Cloud, Hugging Face Spaces, or AWS EC2).
