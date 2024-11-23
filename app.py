import streamlit as st
from src.visualization import (
    plot_feature_importances,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_density,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_3d_predictions,
    plot_classification_report
)
from src.model_training import evaluate_model
from src.data_processing import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load pre-trained models and scaler
@st.cache_resource
def load_models():
    rf_model = joblib.load('models/trained_rf_model.pkl')
    logreg_model = joblib.load('models/trained_logreg_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return rf_model, logreg_model, scaler

# Load and preprocess data
@st.cache_resource
def load_and_prepare_data():
    dataset_path = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prediction_dataset.csv"
    data = load_data(dataset_path)
    features, target, scaler = preprocess_data(data)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(features, target)

    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)
    return data, X_train, X_test, y_train, y_test

st.set_page_config(page_title="Injury Risk Prediction", layout="wide")

# Sidebar
st.sidebar.title("Options")

# Correlation Heatmap button
if st.sidebar.button("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    data, _, _, _, _ = load_and_prepare_data()
    fig = plot_correlation_heatmap(data)
    st.pyplot(fig)

# Model and visualization selection
model_choice = st.sidebar.selectbox("Select a model:", ["Random Forest", "Logistic Regression"])
visualization = st.sidebar.selectbox(
    "Select a visualization:",
    [
        "Classification Report",
        "Confusion Matrix",
        "Density Plot",
        "Feature Importance (Random Forest Only)",
        "Precision-Recall Curve",
        "ROC Curve",
        "Scatterplot",
        "3D Predictions Scatterplot (Logistic Regression Only)",
    ],
)

# Exit button
if st.sidebar.button("Exit"):
    st.stop()

# Load data and models
rf_model, logreg_model, scaler = load_models()
data, X_train, X_test, y_train, y_test = load_and_prepare_data()

# Display visualizations
if visualization == "Classification Report":
    if model_choice == "Random Forest":
        st.subheader("Classification Report (Random Forest)")
        fig = plot_classification_report(rf_model, X_test, y_test, scaler)
        st.pyplot(fig)
    else:
        st.subheader("Classification Report (Logistic Regression)")
        fig = plot_classification_report(logreg_model, X_test, y_test, scaler)
        st.pyplot(fig)

if visualization == "Classification Report":
    if model_choice == "Random Forest":
        st.subheader("Classification Report (Random Forest)")
        st.text(evaluate_model(rf_model, X_test, y_test))  # Display as plain text
    else:
        st.subheader("Classification Report (Logistic Regression)")
        st.text(evaluate_model(logreg_model, X_test, y_test, scaler))

elif visualization == "Confusion Matrix":
    if model_choice == "Random Forest":
        fig = plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
    else:
        fig = plot_confusion_matrix(logreg_model, X_test, y_test, "Logistic Regression")
    st.pyplot(fig)

elif visualization == "Density Plot":
    if model_choice == "Random Forest":
        fig = plot_density(rf_model, X_test, y_test, "Random Forest")
    else:
        fig = plot_density(logreg_model, X_test, y_test, "Logistic Regression")
    st.pyplot(fig)

elif visualization == "Feature Importance (Random Forest Only)":
    if model_choice == "Random Forest":
        feature_names = data.columns[:-1]
        fig = plot_feature_importances(rf_model, feature_names, "Random Forest")
        st.pyplot(fig)
    else:
        st.error("Feature importance is only available for Random Forest.")

elif visualization == "Precision-Recall Curve":
    if model_choice == "Random Forest":
        fig = plot_precision_recall_curve(rf_model, X_test, y_test, "Random Forest")
    else:
        fig = plot_precision_recall_curve(logreg_model, X_test, y_test, "Logistic Regression")
    st.pyplot(fig)

elif visualization == "ROC Curve":
    if model_choice == "Random Forest":
        fig = plot_roc_curve(rf_model, X_test, y_test, "Random Forest")
    else:
        fig = plot_roc_curve(logreg_model, X_test, y_test, "Logistic Regression")
    st.pyplot(fig)

elif visualization == "3D Predictions Scatterplot (Logistic Regression Only)":
    if model_choice == "Logistic Regression":
        feature_names = data.columns[:-1]
        fig = plot_3d_predictions(logreg_model, X_test, y_test, feature_names, "Logistic Regression")
        st.plotly_chart(fig)  # Using Plotly for interactive 3D scatterplot
    else:
        st.error("3D Predictions Scatterplot is only available for Logistic Regression.")
