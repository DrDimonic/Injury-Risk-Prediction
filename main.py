from src.data_processing import load_data, preprocess_data
from src.model_training import train_random_forest, train_logistic_regression, compare_models, evaluate_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.visualization import plot_feature_importances, plot_scatter, plot_3d_predictions, plot_confusion_matrix, plot_density, plot_correlation_heatmap, plot_precision_recall_curve, plot_roc_curve
import joblib
import os

def main():
    # Define paths
    dataset_path = "data/Injury_risk_prediction_dataset.csv"
    rf_model_save_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_rf_model.pkl"
    logreg_model_save_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_logreg_model.pkl"
    scaler_save_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\scaler.pkl"

    # Load the dataset
    print("Loading dataset...")
    data = load_data(dataset_path)
    if data is None:
        print("Dataset could not be loaded.")
        return
    
    # Preprocess the dataset with Standard Scaling
    print("Preprocessing dataset with Standard Scaling...")
    features, target, scaler = preprocess_data(data)

    # Balance the dataset
    print("Balancing the dataset using SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(features, target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

    # Compare Random Forest and Logistic Regression models
    print("\nComparing Random Forest and Logistic Regression models...")
    compare_models(X_train, y_train, X_test, y_test)

    # Train Random Forest model
    print("Training the Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    # Save Random Forest model
    joblib.dump(rf_model, rf_model_save_path)
    print(f"Random Forest model saved to {rf_model_save_path}")

    # Train Logistic Regression model
    print("Training the Logistic Regression model...")
    logreg_model, scaler = train_logistic_regression(X_train, y_train)
    evaluate_model(logreg_model, X_test, y_test, scaler)

    # Save Logistic Regression model and scaler
    joblib.dump(logreg_model, logreg_model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"Logistic Regression model saved to {logreg_model_save_path}")
    print(f"Scaler saved to {scaler_save_path}")

    # Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plot_correlation_heatmap(data) 

    # Visualizations for Random Forest
    print("Generating visualizations for Random Forest...")
    feature_names = data.columns[:-1] 
    plot_feature_importances(rf_model, feature_names, "Random Forest")
    plot_scatter(rf_model, X_test, y_test, "Random Forest") 
    plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
    plot_density(rf_model, X_test, y_test, "Random Forest")
    plot_precision_recall_curve(rf_model, X_test, y_test, "Random Forest")
    plot_roc_curve(rf_model, X_test, y_test, "Random Forest")
    
    # Visualizations for Logistic Regression
    print("Generating visualizations for Logistic Regression...")
    plot_3d_predictions(logreg_model, X_test, y_test, feature_names, "Logistic Regression")
    plot_confusion_matrix(logreg_model, X_test, y_test, "Logistic Regression")
    plot_density(logreg_model, X_test, y_test, "Logistic Regression")
    plot_precision_recall_curve(logreg_model, X_test, y_test, "Logistic Regression")
    plot_roc_curve(logreg_model, X_test, y_test, "Logistic Regression")

if __name__ == "__main__":
    main()
