from src.data_loader import insert_data

def main():
    insert_data("data/Indian_housing_Delhi_data.csv", "Delhi")
    insert_data("data/Indian_housing_Mumbai_data.csv", "Mumbai")
    insert_data("data/Indian_housing_Pune_data.csv", "Pune")

if __name__ == "__main__":
    main()
