import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Plot the feature importances of the model.
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_feature_names, importances[indices])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

# Plot actual vs predicted values.
def plot_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.show()

# Plot a correlation heatmap
def plot_correlation_heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='black'
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# Plot the confusion matrix.
def plot_confusion_matrix(model, X_test, y_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# Overlayed Density plot of actual vs predicted values.
def plot_density(y_test, y_pred_proba):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(y_pred_proba[y_test == 0], label='Class 0', fill=True, alpha=0.5)
    sns.kdeplot(y_pred_proba[y_test == 1], label='Class 1', fill=True, alpha=0.5)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Density Plot of Predicted Probabilities')
    plt.legend()
    plt.show()

