import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from mpl_toolkits.mplot3d import Axes3D

# Plot feature importances
def plot_feature_importances(model, feature_names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_feature_names, importances[indices])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Feature Importances \n({model_name})')
    plt.tight_layout()
    return fig

# Plot correlation heatmap
def plot_correlation_heatmap(data):
    correlation_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='black',
        ax=ax
    )
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    return fig

# Actual vs Predicted values Scatter plot
def plot_scatter(model, X_test, y_test, model_name):
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_test)[:, 1]  
    else:
        y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, c='blue', edgecolors='k')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Scatter Plot: Predicted vs Actual \n({model_name})')
    ax.grid(True)
    plt.tight_layout()
    return fig

# 3D Scatter Plot
def plot_3d_predictions(model, X_test, y_test, feature_names, model_name):
    if len(feature_names) < 2:
        raise ValueError("3D plot requires at least two features.")

    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_test)[:, 1] 
    else:
        y_pred = model.predict(X_test)

    feature_1 = X_test[:, 0] if isinstance(X_test, np.ndarray) else X_test.iloc[:, 0]
    feature_2 = X_test[:, 1] if isinstance(X_test, np.ndarray) else X_test.iloc[:, 1]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(feature_1, feature_2, y_pred, c=y_test, cmap='coolwarm', alpha=0.8, s=50, edgecolor='k')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel('Predicted Probabilities')
    plt.colorbar(scatter, label='Actual Class')
    plt.title(f'3D Scatter Plot: Predictions \n({model_name})')
    plt.tight_layout()
    return fig

# Plot confusion matrix
def plot_confusion_matrix(model, X_test, y_test, model_name):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f'Confusion Matrix \n({model_name})')
    plt.tight_layout()
    return fig

# Overlayed Density plot
def plot_density(model, X_test, y_test, model_name):
    y_pred_proba = model.predict_proba(X_test)
    positive_class_proba = y_pred_proba[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(positive_class_proba[y_test == 0], label='Class 0 (Not Injured)', fill=True, alpha=0.5, color='blue', ax=ax)
    sns.kdeplot(positive_class_proba[y_test == 1], label='Class 1 (Injured)', fill=True, alpha=0.5, color='orange', ax=ax)
    ax.set_xlabel('Predicted Probability for Class 1')
    ax.set_ylabel('Density')
    ax.set_title(f'Density Plot of Predicted Probabilities \n({model_name})')
    ax.legend()
    plt.tight_layout()
    return fig
