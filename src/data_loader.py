import pandas as pd
from .db_connect import get_connection

def insert_data(filepath, city):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    if 'city' not in df.columns:
        df['city'] = city

    df = df.replace({pd.NA: None, 'nan': None, 'NaN': None, '': None})
    df = df.where(pd.notnull(df), None)

    conn = get_connection()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        row = row.to_dict()
        row = {k: (None if pd.isna(v) else v) for k, v in row.items()}

        query = """
        INSERT INTO rental_data (
            house_format, house_type, house_size, location, city, price, numBathrooms, SecurityDeposit, Status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            row.get('house_format'),
            row.get('house_type'),
            row.get('house_size'),
            row.get('location'),
            row.get('city'),
            row.get('price'),
            row.get('numBathrooms'),
            row.get('SecurityDeposit'),
            row.get('Status')
        )

        cursor.execute(query, values)

    conn.commit()
    cursor.close()
    conn.close()
