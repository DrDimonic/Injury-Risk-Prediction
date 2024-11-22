import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot the feature importances from a trained model.
def plot_feature_importances(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        print("The model does not have feature importances.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances", fontsize=16)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90, fontsize=10)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Importance Score", fontsize=14)
    plt.tight_layout()
    plt.show()

# Plot predictions versus actual values.
def plot_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual", alpha=0.6)
    plt.scatter(range(len(predictions)), predictions, color="red", label="Predicted", alpha=0.6)
    plt.title("Predictions vs Actual Values", fontsize=16)
    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("Injury Risk", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# Plot a confusion matrix for the test data.
def plot_confusion_matrix(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix", fontsize=16)
    plt.show()

# Plot a histogram comparing actual and predicted values.
def plot_actual_vs_predicted_histogram(model, X_test, y_test): 
    predictions = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.hist(y_test, bins=10, alpha=0.6, color="blue", label="Actual", edgecolor="black")
    plt.hist(predictions, bins=10, alpha=0.6, color="red", label="Predicted", edgecolor="black")
    plt.title("Actual vs Predicted Histogram", fontsize=16)
    plt.xlabel("Injury Risk", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

