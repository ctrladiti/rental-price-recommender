import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from db_connect import load_data_from_sql

def perform_eda():
    df = load_data_from_sql()
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    # Create folder for plots
    os.makedirs("eda_outputs", exist_ok=True)

    print("\n📄 Data Info:")
    print(df.info())

    print("\n📊 Sample Data:\n", df.head())

    print("\n🧼 Missing Values:\n", df.isnull().sum())

    print("\n🧮 % Missing:\n", round(df.isnull().mean() * 100, 2))

    print("\n📈 Descriptive Statistics:\n", df.describe(include='all'))

    # Categorical Unique Values
    categorical_cols = df.select_dtypes(include='object').columns
    print("\n🔢 Unique Values in Categorical Columns:")
    for col in categorical_cols:
        print(f"{col} → {df[col].nunique()} unique values")
        print("   Example values:", df[col].unique()[:5])

    # Numeric Distributions
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"eda_outputs/{col}_distribution.png")
        plt.close()

    # Count plots for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().iloc[:10].index)
        plt.title(f"Top Categories of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"eda_outputs/{col}_countplot.png")
        plt.close()

    # Boxplot for price by furnishing status
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Status', y='price')
    plt.title("Rental Price by Furnishing Status")
    plt.savefig("eda_outputs/price_by_status.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("eda_outputs/correlation_heatmap.png")
    plt.close()

    # Top locations with most listings
    top_locations = df['location'].value_counts().nlargest(10).index
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df[df['location'].isin(top_locations)],
                x='location', y='price', ci=None)
    plt.xticks(rotation=45)
    plt.title("Avg Price by Top 10 Locations")
    plt.savefig("eda_outputs/avg_price_by_top_locations.png")
    plt.close()

    print("\n✅ EDA completed and visualizations saved in 'eda_outputs/' folder.")

if __name__ == "__main__":
    perform_eda()
