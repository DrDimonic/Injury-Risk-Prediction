from src.data_processing import load_data, preprocess_data
from train import train_model
from evaluate import evaluate_model
import os

# Define the file path for the dataset
#dataset_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prevention_dataset.csv"
dataset_path = os.path.join("data", "Injury_risk_prevention_dataset.csv")


def main():
    # Step 1: Load the dataset
    print("Loading dataset...")
    data = load_data(dataset_path)
    if data is None:
        print("Failed to load the dataset.")
        return

    print("Dataset loaded successfully!")

    # Step 2: Preprocess the dataset
    print("Preprocessing dataset...")
    try:
        features, target = preprocess_data(data)
    except KeyError as e:
        print(f"Error during preprocessing: {e}")
        return

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()