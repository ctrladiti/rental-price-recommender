import pandas as pd
import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='rental_data'
    )

def load_data_from_sql():
    conn = get_connection()
    query = "SELECT * FROM rental_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_table_as_dataframe(table_name: str) -> pd.DataFrame:
    connection = get_connection()
    query = f"SELECT * FROM {table_name};"
    
    print(f"📦 Executing query: {query}")
    df = pd.read_sql(query, connection)

    print(f"✅ Loaded from SQL: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

