import pandas as pd
from db_connect import get_connection

def clean_column_names(df):
    return df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))

def insert_data(file_path, city_name):
    conn = get_connection()
    cursor = conn.cursor()

    df = pd.read_csv(file_path)
    df = clean_column_names(df)
    df['city'] = city_name

    insert_query = """
    INSERT INTO listings (area, bhk, size_sqft, bathroom, furnished_status, rent, city)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    for _, row in df.iterrows():
        cursor.execute(insert_query, (
            row['area'], row['bhk'], row['size'], row['bathroom'],
            row.get('furnishing_status', 'Unknown'), row['rent'], row['city']
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Data from {file_path} inserted successfully.")

