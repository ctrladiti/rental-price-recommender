import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: For better plots
sns.set(style="whitegrid")

def run_eda(filepath):
    df = pd.read_csv(filepath)

    # Basic info
    print("\n--- Dataset Info ---")
    print(df.info())

    # Null values
    print("\n--- Null Values ---")
    print(df.isnull().sum())

    # Describe numeric columns
    print("\n--- Statistical Summary ---")
    print(df.describe())

    # Unique values for categorical columns
    print("\n--- Unique Categorical Values ---")
    for col in ['house_type', 'location', 'Status']:
        if col in df.columns:
            print(f"{col}: {df[col].unique()}")

    # Correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("eda_outputs/correlation_matrix.png")

    # Price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['price'], kde=True, bins=30)
    plt.title("Price Distribution")
    plt.xlabel("Rental Price")
    plt.tight_layout()
    plt.savefig("eda_outputs/price_distribution.png")

    # Price vs House Size
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='house_size', y='price', hue='house_type')
    plt.title("Price vs House Size")
    plt.tight_layout()
    plt.savefig("eda_outputs/price_vs_size.png")

    # Location-wise average price
    if 'location' in df.columns:
        top_locations = df['location'].value_counts().nlargest(10).index
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df[df['location'].isin(top_locations)],
                    x='location', y='price', estimator='mean')
        plt.xticks(rotation=45)
        plt.title("Average Price by Top 10 Locations")
        plt.tight_layout()
        plt.savefig("eda_outputs/avg_price_by_location.png")

    print("\nEDA plots saved to the 'eda_outputs/' directory.")

# Run EDA (You can call this from main.py if needed)
if __name__ == "__main__":
    run_eda("data/Indian_housing_Delhi_data.csv")
