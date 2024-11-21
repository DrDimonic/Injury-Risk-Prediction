from src.data_processing import load_dataset, preprocess_dataset
import os

# Define the file path for the dataset
dataset_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prevention_dataset.csv"

# Load the dataset
data = load_dataset(dataset_path)

if data is not None:
    # Preprocess the dataset
    processed_data = preprocess_dataset(data)
    print("First 5 rows of the processed dataset:")
    print(processed_data.head())

    # Optional: Save the processed data to a new file for debugging
    processed_data.to_csv(r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\processed_players_injury_data.csv", index=False)
else:
    print("Failed to load the dataset.")
