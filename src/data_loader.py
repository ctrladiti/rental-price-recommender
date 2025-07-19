import pandas as pd
from .db_connect import get_connection

def insert_data(filepath, city):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # in case of whitespaces
    df = df.replace({pd.NA: None, 'nan': None, 'NaN': None, '': None})
    df = df.where(pd.notnull(df), None)  # convert NaN to None

    conn = get_connection()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        query = """
        INSERT INTO rental_data (
            house_type, house_size, location, city, latitude, longitude,
            price, currency, numBathrooms, numBalconies, isNegotiable,
            priceSqFt, verificationDate, description, SecurityDeposit, Status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            row['house_type'],
            row['house_size'],
            row['location'],
            row['city'],
            row['latitude'],
            row['longitude'],
            row['price'],
            row['currency'],
            row['numBathrooms'],
            row['numBalconies'],
            row['isNegotiable'],
            row['priceSqFt'],
            row['verificationDate'],
            row['description'],
            row['SecurityDeposit'],
            row['Status']
        )

        cursor.execute(query, values)

    conn.commit()
    cursor.close()
    conn.close()
