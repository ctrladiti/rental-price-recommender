import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Optional: For better plots
sns.set(style="whitegrid")

def run_eda(filepath, city):
    # Create output directory if not exists
    os.makedirs("eda_outputs", exist_ok=True)

    # Load dataset
    df = pd.read_csv(filepath)

    # Basic info
    print(f"\n--- Dataset Info for {city} ---")
    print(df.info())

    # Null values
    print(f"\n--- Null Values in {city} ---")
    print(df.isnull().sum())

    # Describe numeric columns
    print(f"\n--- Statistical Summary for {city} ---")
    print(df.describe())

    # Unique values for categorical columns
    print(f"\n--- Unique Categorical Values in {city} ---")
    for col in ['house_type', 'location', 'Status']:
        if col in df.columns:
            print(f"{col}: {df[col].unique()}")

    # Convert house_size to numeric if needed
    if 'house_size' in df.columns and df['house_size'].dtype == 'object':
        df['house_size'] = df['house_size'].str.replace(r"[^\d.]", "", regex=True).astype(float)

    # Correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"eda_outputs/{city}_correlation_matrix.png")

    # Price distribution
    if 'price' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df['price'], kde=True, bins=30)
        plt.title("Price Distribution")
        plt.xlabel("Rental Price")
        plt.tight_layout()
        plt.savefig(f"eda_outputs/{city}_price_distribution.png")

    # Price vs House Size
    if 'house_size' in df.columns and 'price' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='house_size', y='price', hue='house_type')
        plt.title("Price vs House Size")
        plt.tight_layout()
        plt.savefig(f"eda_outputs/{city}_price_vs_size.png")

    # Location-wise average price
    if 'location' in df.columns and 'price' in df.columns:
        top_locations = df['location'].value_counts().nlargest(10).index
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df[df['location'].isin(top_locations)],
                    x='location', y='price', estimator='mean', ci=None)
        plt.xticks(rotation=45)
        plt.title("Average Price by Top 10 Locations")
        plt.tight_layout()
        plt.savefig(f"eda_outputs/{city}_avg_price_by_location.png")

    print(f"\nEDA plots for {city} saved to the 'eda_outputs/' directory.")

# Run EDA (You can call this from main.py if needed)
if __name__ == "__main__":
    run_eda("data/Indian_housing_Delhi_data.csv", "Delhi")
    run_eda("data/Indian_housing_Mumbai_data.csv", "Mumbai")
    run_eda("data/Indian_housing_Pune_data.csv", "Pune")
