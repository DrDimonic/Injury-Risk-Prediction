import streamlit as st
import joblib
from src.data_processing import load_data, preprocess_data
from src.visualization import (
    plot_feature_importances,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_scatter,
    plot_3d_predictions,
    plot_density,
    plot_precision_recall_curve,
    plot_roc_curve
)
from src.model_training import train_random_forest, train_logistic_regression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Preload dataset and preprocess
DATASET_PATH = r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\Injury_risk_prediction_dataset.csv"
data = load_data(DATASET_PATH)
features, target, scaler = preprocess_data(data)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(features, target)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Train models
rf_model = train_random_forest(X_train, y_train)
logreg_model, logreg_scaler = train_logistic_regression(X_train, y_train)

# Streamlit interface
st.sidebar.title("Injury Risk Prediction")
st.sidebar.subheader("Choose a Model")
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"])

if model_choice == "Random Forest":
    st.header("Random Forest Visualizations")
    feature_names = data.columns[:-1]

    if st.button("Feature Importances"):
        fig = plot_feature_importances(rf_model, feature_names, "Random Forest")
        st.pyplot(fig)

    if st.button("Scatter Plot"):
        fig = plot_scatter(rf_model, X_test, y_test, "Random Forest")
        st.pyplot(fig)

    if st.button("3D Scatter Plot"):
        fig = plot_3d_predictions(rf_model, X_test, y_test, feature_names, "Random Forest")
        st.pyplot(fig)

    if st.button("Correlation Heatmap"):
        fig = plot_correlation_heatmap(data)
        st.pyplot(fig)

elif model_choice == "Logistic Regression":
    st.header("Logistic Regression Visualizations")
    feature_names = data.columns[:-1]

    if st.button("Scatter Plot"):
        fig = plot_scatter(logreg_model, X_test, y_test, "Logistic Regression")
        st.pyplot(fig)

    if st.button("3D Scatter Plot"):
        fig = plot_3d_predictions(logreg_model, X_test, y_test, feature_names, "Logistic Regression")
        st.pyplot(fig)

    if st.button("Correlation Heatmap"):
        fig = plot_correlation_heatmap(data)
        st.pyplot(fig)

# Exit button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Exit"):
    st.stop()
