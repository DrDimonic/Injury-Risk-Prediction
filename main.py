from src.data_processing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model
from src.visualization import plot_feature_importances, plot_predictions, plot_confusion_matrix, plot_actual_vs_predicted_histogram
from sklearn.model_selection import train_test_split
import joblib
import os

# Define the dataset file path.
#dataset_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prevention_dataset.csv"
dataset_path = os.path.join("data", "Injury_risk_prevention_dataset.csv")
model_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_model.pkl"


def main():
   # Load and preprocess the data
    print("Loading dataset...")
    data = load_data(dataset_path)
    if data is None:
        print("Failed to load the dataset.")
        return

    print("Preprocessing dataset...")
    features, target = preprocess_data(data)

    # Train the model
    print("Training the model...")
    model, X_test, y_test = train_model(features, target)
    print("Model training complete.")

    # Ensure the 'models' directory exists
    models_dir = os.path.dirname(model_path)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved as '{model_path}'")

    # Visualizations
    print("Generating visualizations...")
    feature_names = features.columns if hasattr(features, "columns") else [f"Feature {i}" for i in range(features.shape[1])]
    plot_feature_importances(model, feature_names)
    plot_predictions(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test) 
    plot_actual_vs_predicted_histogram(model, X_test, y_test)

if __name__ == "__main__":
    main()