from src.data_processing import load_data, preprocess_data
from src.model_training import train_random_forest, train_logistic_regression, compare_models
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.visualization import plot_feature_importances, plot_bubble_chart, plot_3d_predictions, plot_confusion_matrix, plot_density, plot_correlation_heatmap
from sklearn.linear_model import LogisticRegression
import joblib
import os

def main():
    # Define paths
    dataset_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prediction_dataset.csv"
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

    # Database Visualizations 
    print("Generating Correlation Heatmap...")
    plot_correlation_heatmap(data)


    # Visualizations for Random Forest
    print("Generating visualizations for Random Forest...")
    feature_names = data.columns[:-1] 
    plot_feature_importances(rf_model, feature_names)
    plot_bubble_chart(rf_model, X_test, y_test)   
    plot_confusion_matrix(rf_model, X_test, y_test)
    plot_density(rf_model, X_test, y_test)

    # Visualizations for Logistic Regression
    print("Generating visualizations for Logistic Regression...")
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    plot_3d_predictions(logreg_model, X_test, y_test)
    plot_confusion_matrix(logreg, X_test, y_test)
    plot_density(logreg, X_test, y_test)

if __name__ == "__main__":
    main()
