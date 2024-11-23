from src.data_processing import load_data, preprocess_data
from src.model_training import train_random_forest, train_logistic_regression, evaluate_model, compare_models
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def main():
    # Define paths
    dataset_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prevention.csv"
    rf_model_save_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_rf_model.pkl"
    logreg_model_save_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_logreg_model.pkl"
    scaler_save_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\scaler.pkl"

    # Load and preprocess the dataset
    print("Loading dataset...")
    data = load_data(dataset_path)
    if data is None:
        print("Dataset could not be loaded.")
        return

    features, target = preprocess_data(data)

    # Balance the dataset
    print("Balancing the dataset using SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(features, target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

    # Compare Random Forest and Logistic Regression models
    print("\nComparing Random Forest and Logistic Regression models...")
    compare_models(X_train, y_train, X_test, y_test)

    # Train and save models individually
    print("\nSaving models...")
    rf_model = train_random_forest(X_train, y_train)
    joblib.dump(rf_model, rf_model_save_path)
    print(f"Random Forest model saved to {rf_model_save_path}")

    logreg_model, scaler = train_logistic_regression(X_train, y_train)
    joblib.dump(logreg_model, logreg_model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"Logistic Regression model saved to {logreg_model_save_path}")
    print(f"Scaler saved to {scaler_save_path}")

if __name__ == "__main__":
    main()
