from src.data_processing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model
import joblib
import os

def main():
    # Define dataset and model save paths
    dataset_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\dataset.csv"
    model_save_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_model.pkl"

    # Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    data = load_data(dataset_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train the model
    print("Training the model...")
    model = train_model(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Save the model
    joblib.dump(model, model_save_path)
    print(f"Model saved as '{model_save_path}'")

if __name__ == "__main__":
    main()
