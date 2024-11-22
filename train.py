from src.data_processing import load_data, preprocess_data
from src.model_training import train_random_forest, train_logistic_regression, evaluate_model
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

def main():
    # Define dataset and model save paths
    dataset_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\dataset.csv"
    model_save_path_rf = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_rf_model.pkl"
    model_save_path_logreg = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_logreg_model.pkl"

    # Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    data = load_data(dataset_path)
    if data is None:
        print("Dataset could not be loaded.")
        return

    # Split features and target
    features, target = preprocess_data(data)

    # Apply SMOTE for balancing the dataset
    print("Applying SMOTE for balancing the dataset...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(features, target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

    # Train Random Forest model
    print("Training the Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    print("Evaluating the Random Forest model...")
    evaluate_model(rf_model, X_test, y_test)

    # Save the Random Forest model
    joblib.dump(rf_model, model_save_path_rf)
    print(f"Random Forest model saved as '{model_save_path_rf}'")

    # Train Logistic Regression model
    print("Training the Logistic Regression model...")
    logreg_model, scaler = train_logistic_regression(X_train, y_train)
    print("Evaluating the Logistic Regression model...")
    evaluate_model(logreg_model, X_test, y_test, scaler)

    # Save the Logistic Regression model and scaler
    joblib.dump(logreg_model, model_save_path_logreg)
    print(f"Logistic Regression model saved as '{model_save_path_logreg}'")
    scaler_save_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\scaler.pkl"
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved as '{scaler_save_path}'")

if __name__ == "__main__":
    main()
