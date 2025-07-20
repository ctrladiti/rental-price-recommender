from db_connect import load_data_from_sql

def perform_eda():
    df = load_data_from_sql()
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    print("\n📊 Sample Data:\n", df.head())
    print("\n🧼 Missing Values:\n", df.isnull().sum())
    print("\n📈 Descriptive Stats:\n", df.describe(include='all'))


if __name__ == "__main__":
    perform_eda()
