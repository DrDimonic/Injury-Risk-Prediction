from src.data_processing import load_data, preprocess_data
from src.model_training import train_random_forest, train_logistic_regression, evaluate_model, cross_validate_model, compare_models
from src.visualization import plot_feature_importances, plot_predictions, plot_confusion_matrix, plot_actual_vs_predicted_histogram, plot_correlation_heatmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import os
import joblib

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
    
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Perform cross-validation
    print("Performing cross-validation...")
    cross_validate_model(X_train, y_train)

    # Train the Random Forest model
    print("Training the Random Forest model...")
    model = train_model(X_train, y_train)

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved as '{model_path}'")

    # Evaluate the model
    print("Evaluating the Random Forest model...")
    evaluate_model(model, X_test, y_test)

    # Compare models
    print("Comparing models...")
    compare_models(X_train, y_train, X_test, y_test)

    # Visualizations for Random Forest
    print("Generating visualizations for Random Forest...")
    feature_names = data.columns[:-1] 
    plot_feature_importances(model, feature_names)
    plot_correlation_heatmap(data)
    plot_predictions(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_actual_vs_predicted_histogram(model, X_test, y_test)

    # Visualizations for Logistic Regression
    print("Generating visualizations for Logistic Regression...")
    logreg = LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42)
    logreg.fit(X_train, y_train)
    plot_feature_importances(model, feature_names)
    plot_correlation_heatmap(data)
    plot_predictions(logreg, X_test, y_test)
    plot_confusion_matrix(logreg, X_test, y_test)
    plot_actual_vs_predicted_histogram(logreg, X_test, y_test)

if __name__ == "__main__":
    main()


