from src.data_processing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model
from src.visualization import plot_feature_importances, plot_predictions, plot_correlation_heatmap, plot_confusion_matrix, plot_actual_vs_predicted_histogram
from sklearn.model_selection import train_test_split
import joblib
import os

# Define the dataset file path.
#dataset_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prevention_dataset.csv"
dataset_path = os.path.join("data", "Injury_risk_prevention_dataset.csv")
model_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_model.pkl"


def main():
   # Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    data = load_data(dataset_path)
    if data is None:
        print("Dataset could not be loaded.")
        return
    
    features, target = preprocess_data(data)

    # Train the model
    print("Training the model...")
    model = train_model(features, target)
    print("Model training complete.")

    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Visualizations
    print("Generating visualizations...")
    feature_names = features.columns if hasattr(features, "columns") else [f"Feature {i}" for i in range(features.shape[1])]
    plot_feature_importances(model, feature_names)
    plot_predictions(model, X_test, y_test)
    plot_correlation_heatmap(data)
    plot_confusion_matrix(model, X_test, y_test) 
    plot_actual_vs_predicted_histogram(model, X_test, y_test)


if __name__ == "__main__":
    main()


