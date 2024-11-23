import streamlit as st
import pandas as pd
from src.data_processing import preprocess_data
from src.model_training import train_random_forest, train_logistic_regression, evaluate_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.visualization import plot_feature_importances, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_density

# Preload dataset path
DATASET_PATH = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prediction_dataset.csv"

# Load and preprocess the dataset
@st.cache
def load_preprocessed_data():
    data = pd.read_csv(DATASET_PATH)
    features, target, scaler = preprocess_data(data)
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(features, target)
    return train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42), scaler

def main():
    st.title("Injury Risk Prediction")

    # Load preprocessed data
    (X_train, X_test, y_train, y_test), scaler = load_preprocessed_data()

    # Train Random Forest
    st.subheader("Random Forest Training")
    rf_model = train_random_forest(X_train, y_train)
    st.write("Random Forest model trained successfully!")

    # Train Logistic Regression
    st.subheader("Logistic Regression Training")
    logreg_model, _ = train_logistic_regression(X_train, y_train)
    st.write("Logistic Regression model trained successfully!")

    # Random Forest Visualizations
    st.subheader("Random Forest Visualizations")
    st.write("Feature Importances")
    fig_importance = plot_feature_importances(rf_model, X_train.columns, "Random Forest")
    st.pyplot(fig_importance)

    st.write("Confusion Matrix")
    fig_confusion_rf = plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
    st.pyplot(fig_confusion_rf)

    st.write("Precision-Recall Curve")
    fig_pr_curve = plot_precision_recall_curve(rf_model, X_test, y_test, "Random Forest")
    st.pyplot(fig_pr_curve)

    # Logistic Regression Visualizations
    st.subheader("Logistic Regression Visualizations")
    st.write("Density Plot")
    fig_density = plot_density(logreg_model, X_test, y_test, "Logistic Regression")
    st.pyplot(fig_density)

    st.write("ROC Curve")
    fig_roc_curve = plot_roc_curve(logreg_model, X_test, y_test, "Logistic Regression")
    st.pyplot(fig_roc_curve)

if __name__ == "__main__":
    main()
