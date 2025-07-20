import pandas as pd
from src.data_loader import insert_data


def main():
    # df = pd.read_csv("data/Indian_housing_Delhi_data.csv")
    # print(df)
    # insert_data("data/Indian_housing_Delhi_data.csv", "Delhi")
    insert_data("data/Indian_housing_Mumbai_data.csv", "Mumbai")
    insert_data("data/Indian_housing_Pune_data.csv", "Pune")

if __name__ == "__main__":
    main()