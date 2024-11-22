from src.data_processing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model, cross_validate_model, compare_models
from src.visualization import plot_feature_importances, plot_predictions, plot_confusion_matrix, plot_actual_vs_predicted_histogram, plot_correlation_heatmap
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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
    
    features, target = preprocess_data(data)

    # Balance the data
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(features, target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

    # Perform cross-validation
    print("Performing cross-validation...")
    cross_validate_model(features, target)

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
    feature_names = features.columns if hasattr(features, "columns") else [f"Feature {i}" for i in range(features.shape[1])]
    plot_feature_importances(model, feature_names)
    plot_predictions(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_actual_vs_predicted_histogram(model, X_test, y_test)

    # Visualizations for Logistic Regression
    print("Generating visualizations for Logistic Regression...")
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    plot_predictions(logreg, X_test, y_test)
    plot_confusion_matrix(logreg, X_test, y_test)
    plot_actual_vs_predicted_histogram(logreg, X_test, y_test)


if __name__ == "__main__":
    main()


