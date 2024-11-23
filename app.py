import streamlit as st
from src.visualization import (
    plot_feature_importances,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_density,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_scatter,
    plot_3d_predictions
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

# Streamlit App
st.set_page_config(page_title="Injury Risk Prediction", layout="wide")

# Sidebar Options
st.sidebar.title("Options")
visualization = st.sidebar.selectbox(
    "Select a visualization or report:",
    [
        "Classification Report (Logistic Regression)",
        "Classification Report (Random Forest)",
        "Confusion Matrix",
        "Correlation Heatmap",
        "Density Plot",
        "Feature Importance (Random Forest Only)",
        "Precision-Recall Curve",
        "ROC Curve",
        "3D Predictions Scatterplot (Logistic Regression Only)"
    ]
)

# Exit button at the bottom of the sidebar
if st.sidebar.button("Exit"):
    st.stop()

# Load models and data
rf_model, logreg_model, scaler = load_models()
data, X_train, X_test, y_train, y_test = load_and_prepare_data()

# Display selected visualization
if visualization == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    fig = plot_correlation_heatmap(data)
    st.pyplot(fig)

elif visualization == "Feature Importance (Random Forest Only)":
    st.subheader("Feature Importance (Random Forest Only)")
    feature_names = data.columns[:-1]
    fig = plot_feature_importances(rf_model, feature_names, "Random Forest")
    st.pyplot(fig)

elif visualization == "3D Predictions Scatterplot (Logistic Regression Only)":
    st.subheader("3D Predictions Scatterplot (Logistic Regression Only)")
    feature_names = data.columns[:-1]
    fig = plot_3d_predictions(logreg_model, X_test, y_test, feature_names, "Logistic Regression")
    st.pyplot(fig)

elif visualization == "Confusion Matrix":
    st.subheader("Confusion Matrix")
    fig_rf = plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
    st.pyplot(fig_rf)

elif visualization == "Density Plot":
    st.subheader("Density Plot")
    fig_rf = plot_density(rf_model, X_test, y_test, "Random Forest")
    st.pyplot(fig_rf)

elif visualization == "Precision-Recall Curve":
    st.subheader("Precision-Recall Curve")
    fig_rf = plot_precision_recall_curve(rf_model, X_test, y_test, "Random Forest")
    st.pyplot(fig_rf)

elif visualization == "ROC Curve":
    st.subheader("ROC Curve")
    fig_rf = plot_roc_curve(rf_model, X_test, y_test, "Random Forest")
    st.pyplot(fig_rf)

elif visualization == "Classification Report (Random Forest)":
    st.subheader("Classification Report (Random Forest)")
    with st.expander("Random Forest Classification Report"):
        st.text(evaluate_model(rf_model, X_test, y_test))

elif visualization == "Classification Report (Logistic Regression)":
    st.subheader("Classification Report (Logistic Regression)")
    with st.expander("Logistic Regression Classification Report"):
        st.text(evaluate_model(logreg_model, X_test, y_test, scaler))
