import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D



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

# Bubble chart for actual vs predicted values.
def plot_bubble_chart(model, X_test, y_test):
    # Predict probabilities or labels
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_test)[:, 1]  # Use probabilities for the positive class
    else:
        y_pred = model.predict(X_test)

# Actual vs Predicted values Scatter plot
def plot_scatter(model, X_test, y_test):
   
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_test)[:, 1]  # Use probabilities for positive class
    else:
        y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, c='blue', edgecolors='k')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot: Predicted vs Actual')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_3d_predictions(model, X_test, y_test, feature_names):
    # Ensure valid features for plotting
    if len(feature_names) < 2:
        print("3D plot requires at least two features.")
        return

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_test)[:, 1] 
    else:
        y_pred = model.predict(X_test)

    # Extract first two features
    feature_1 = X_test[:, 0] if isinstance(X_test, np.ndarray) else X_test.iloc[:, 0]
    feature_2 = X_test[:, 1] if isinstance(X_test, np.ndarray) else X_test.iloc[:, 1]

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(feature_1, feature_2, y_pred, c=y_test, cmap='coolwarm', alpha=0.8, s=50, edgecolor='k')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel('Predicted Probabilities')
    plt.colorbar(scatter, label='Actual Class')
    plt.title('3D Scatter Plot: Predictions')
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
def plot_density(model, X_test, y_test):
    # Predict and extract probabilites
    y_pred_proba = model.predict_proba(X_test)
    positive_class_proba = y_pred_proba[:, 1]

    # Overlayed density plot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(positive_class_proba[y_test == 0], label='Class 0 (Not Injured)', fill=True, alpha=0.5, color='blue')
    sns.kdeplot(positive_class_proba[y_test == 1], label='Class 1 (Injured)', fill=True, alpha=0.5, color='orange')
    plt.xlabel('Predicted Probability for Class 1')
    plt.ylabel('Density')
    plt.title('Density Plot of Predicted Probabilities')
    plt.legend()
    plt.tight_layout()
    plt.show()

