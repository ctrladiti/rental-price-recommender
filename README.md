# 🏠 Rental Price Recommendation System for Urban Areas

This project aims to analyze and predict rental prices across major Indian metropolitan cities using real-world housing datasets. It integrates **MySQL** for data storage, **Python (Pandas & Scikit-learn)** for processing and modeling, and supports a modular and scalable architecture.

---

## 📁 Project Structure

```
rental-price-recommendation/
│
├── data/
│   ├── Indian_housing_Delhi_data.csv
│   ├── Indian_housing_Mumbai_data.csv
│   └── Indian_housing_Pune_data.csv
│
├── db/
│   ├── rental_data.sql
│   └── Local MYSQL Rental.session.sql
│
├── src/
│   ├── db_connect.py
│   └── data_loader.py
│
├── main.py
└── README.md
```

---

## ✅ Features

- 🚪 Loads multi-city housing rental data from CSVs.
- 🛢️ Stores structured data into MySQL using a normalized schema.
- 📊 Enables city-wise and area-wise rent analysis.
- 🧠 Ready for machine learning modeling (e.g., linear regression, decision trees).
- 🔗 Modular and easy to extend for more cities or data sources.

---

## 🗂️ Dataset Used

Combined rental listing datasets from three Indian cities:

- Delhi
- Mumbai
- Pune

Each CSV contains fields like area, BHK, size in sqft, bathrooms, furnishing status, and rent.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rental-price-recommendation.git
cd rental-price-recommendation
```

### 2. Create MySQL Database

Run the following in VS Code SQLTools or MySQL CLI:

```sql
SOURCE db/rental_data.sql;
```

This creates a database `rental_data` and a table `listings`.

### 3. Configure DB Connection

Create `src/db_connect.py` with your MySQL credentials:

```python
import mysql.connector

def connect():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_password",
        database="rental_data"
    )
```

### 4. Install Dependencies

```bash
pip install pandas mysql-connector-python
```

---

## 🚀 Load Data

Run the loader script to populate MySQL:

```bash
python main.py
```

This will load all 3 datasets into the `listings` table.

---

## 📈 Sample Analysis Queries

```sql
-- Top 5 most expensive areas (avg rent)
SELECT city, area, ROUND(AVG(rent), 2) as avg_rent
FROM listings
GROUP BY city, area
ORDER BY avg_rent DESC
LIMIT 5;

-- Furnishing impact
SELECT furnished_status, ROUND(AVG(rent), 2) as avg_rent
FROM listings
GROUP BY furnished_status;
```

---

## 🔮 Future Enhancements

- Predict rental prices using machine learning models
- Add filtering by amenities and location proximity
- Web dashboard for visualization
- API for real-time price recommendation

---

## 👩‍💻 Author

**Aditi Agrawal** <br>
GitHub: [Aditi-1304](https://github.com/Aditi-1304)
