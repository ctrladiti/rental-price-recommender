import pandas as pd
from src.db_connect import get_connection

def read_all_data_from_sql():
    connection = get_connection()
    query = "SELECT * FROM rental_data"
    df = pd.read_sql(query, connection)
    connection.close()
    return df
