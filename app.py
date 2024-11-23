import streamlit as st
from src.data_processing import load_data, preprocess_data
from src.model_training import train_random_forest, train_logistic_regression, evaluate_model
from src.visualization import (
    plot_feature_importances,
    plot_scatter,
    plot_3d_predictions,
    plot_confusion_matrix,
    plot_density,
    plot_correlation_heatmap,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import io
import os

# Path definitions
DATASET_PATH = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prediction_dataset.csv"

# Preload dataset and preprocess
st.title("Injury Risk Prediction Dashboard")
st.sidebar.header("Model Options")

@st.cache_resource
def load_and_preprocess():
    data = load_data(DATASET_PATH)
    features, target, scaler = preprocess_data(data)
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(features, target)
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, data, scaler

X_train, X_test, y_train, y_test, data, scaler = load_and_preprocess()

# Train models
@st.cache_resource
def train_models():
    rf_model = train_random_forest(X_train, y_train)
    logreg_model, logreg_scaler = train_logistic_regression(X_train, y_train)
    return rf_model, logreg_model, logreg_scaler

rf_model, logreg_model, logreg_scaler = train_models()

# Sidebar options
model_choice = st.sidebar.selectbox("Select a Model", ("Random Forest", "Logistic Regression"))
visualization_choice = st.sidebar.multiselect(
    "Choose Visualizations",
    [
        "Feature Importances (Random Forest only)",
        "Scatter Plot",
        "3D Scatter Plot (Logistic Regression only)",
        "Confusion Matrix",
        "Density Plot",
        "Precision-Recall Curve",
        "ROC Curve",
    ],
)

# Button for correlation heatmap
if st.sidebar.button("Generate Correlation Heatmap"):
    st.write("#### Correlation Heatmap")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plot_correlation_heatmap(data)
    st.pyplot(fig)

# Display selected visualizations and scores
if st.sidebar.button("Generate Visualizations"):
    st.write(f"### {model_choice} Visualizations")

    if "Feature Importances (Random Forest only)" in visualization_choice and model_choice == "Random Forest":
        st.write("#### Feature Importances")
        fig = plt.figure(figsize=(10, 6))
        plot_feature_importances(rf_model, data.columns[:-1], "Random Forest")
        st.pyplot(fig)

    if "Scatter Plot" in visualization_choice:
        st.write("#### Scatter Plot")
        fig = plt.figure(figsize=(10, 6))
        plot_scatter(rf_model if model_choice == "Random Forest" else logreg_model, X_test, y_test, model_choice)
        st.pyplot(fig)

    if "3D Scatter Plot (Logistic Regression only)" in visualization_choice and model_choice == "Logistic Regression":
        st.write("#### 3D Scatter Plot")
        fig = plt.figure(figsize=(10, 6))
        plot_3d_predictions(logreg_model, X_test, y_test, data.columns[:-1], "Logistic Regression")
        st.pyplot(fig)

    if "Confusion Matrix" in visualization_choice:
        st.write("#### Confusion Matrix")
        fig = plt.figure(figsize=(8, 6))
        plot_confusion_matrix(rf_model if model_choice == "Random Forest" else logreg_model, X_test, y_test, model_choice)
        st.pyplot(fig)

    if "Density Plot" in visualization_choice:
        st.write("#### Density Plot")
        fig = plt.figure(figsize=(8, 5))
        plot_density(rf_model if model_choice == "Random Forest" else logreg_model, X_test, y_test, model_choice)
        st.pyplot(fig)

    if "Precision-Recall Curve" in visualization_choice:
        st.write("#### Precision-Recall Curve")
        fig = plt.figure(figsize=(8, 6))
        plot_precision_recall_curve(rf_model if model_choice == "Random Forest" else logreg_model, X_test, y_test, model_choice)
        st.pyplot(fig)

    if "ROC Curve" in visualization_choice:
        st.write("#### ROC Curve")
        fig = plt.figure(figsize=(8, 6))
        plot_roc_curve(rf_model if model_choice == "Random Forest" else logreg_model, X_test, y_test, model_choice)
        st.pyplot(fig)

# Display evaluation metrics
if st.sidebar.button("Display Scores"):
    st.write(f"### {model_choice} Evaluation Metrics")
    output = io.StringIO()
    if model_choice == "Random Forest":
        evaluate_model(rf_model, X_test, y_test)
    else:
        evaluate_model(logreg_model, X_test, y_test, logreg_scaler)
