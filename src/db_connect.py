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
