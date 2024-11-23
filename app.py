import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data_processing import preprocess_data
from src.model_training import train_random_forest, train_logistic_regression, evaluate_model
from src.visualization import (
    plot_confusion_matrix,
    plot_density,
    plot_correlation_heatmap,
    plot_precision_recall_curve,
    plot_roc_curve,
)

# Title and Description
st.title("Injury Risk Prediction Application")
st.markdown("""
This application predicts the injury risk for athletes based on various features.
You can upload your dataset, preprocess the data, train models, evaluate them, and visualize results.
""")

# File Upload Section
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Load and display dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    # Preprocessing Section
    st.subheader("Preprocess the Dataset")
    features, target, scaler = preprocess_data(data)
    st.write("Features Preview:", pd.DataFrame(features).head())
    st.write("Target Distribution:", target.value_counts())

    # Balance Dataset with SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(features, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=42
    )
    st.write("Dataset balanced with SMOTE.")

    # Model Training Section
    st.subheader("Train Models")

    # Train Random Forest
    if st.button("Train Random Forest"):
        rf_model = train_random_forest(X_train, y_train)
        joblib.dump(rf_model, "trained_rf_model.pkl")
        st.success("Random Forest model trained and saved!")
        st.write("Random Forest Evaluation:")
        evaluate_model(rf_model, X_test, y_test)

    # Train Logistic Regression
    if st.button("Train Logistic Regression"):
        logreg_model, scaler = train_logistic_regression(X_train, y_train)
        joblib.dump(logreg_model, "trained_logreg_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("Logistic Regression model trained and saved!")
        st.write("Logistic Regression Evaluation:")
        evaluate_model(logreg_model, X_test, y_test, scaler)

    # Visualizations Section
    st.subheader("Visualizations")
    if st.button("Generate Visualizations"):
        st.write("Correlation Heatmap:")
        plot_correlation_heatmap(data)
        st.pyplot()

        st.write("Confusion Matrix for Random Forest:")
        plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
        st.pyplot()

        st.write("Density Plot for Logistic Regression:")
        plot_density(logreg_model, X_test, y_test, "Logistic Regression")
        st.pyplot()

        st.write("Precision-Recall Curve for Random Forest:")
        plot_precision_recall_curve(rf_model, X_test, y_test, "Random Forest")
        st.pyplot()

        st.write("ROC Curve for Logistic Regression:")
        plot_roc_curve(logreg_model, X_test, y_test, "Logistic Regression")
        st.pyplot()

# Prediction Section
st.subheader("Make a Prediction")
if uploaded_file is not None:
    user_input = {}
    for col in data.columns[:-1]:  # Exclude the target column
        user_input[col] = st.number_input(f"Enter {col}:", value=0.0)

    if st.button("Predict"):
        user_df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(user_df)
        rf_prediction = rf_model.predict(scaled_input)[0]
        logreg_prediction = logreg_model.predict(scaled_input)[0]

        st.write(f"Random Forest Prediction: {'Injured' if rf_prediction else 'Not Injured'}")
        st.write(f"Logistic Regression Prediction: {'Injured' if logreg_prediction else 'Not Injured'}")
