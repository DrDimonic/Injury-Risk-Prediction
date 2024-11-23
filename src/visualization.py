import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D  # For fallback Matplotlib 3D plots

# Global figure counter
figure_counter = 1

def increment_figure_counter():
    global figure_counter
    figure_counter += 1

def get_figure_number():
    return figure_counter

# Plot Correlation Heatmap
def plot_correlation_heatmap(data):
    global figure_counter
    correlation_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjusted size
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
    increment_figure_counter()
    return fig

# Plot Confusion Matrix
def plot_confusion_matrix(model, X_test, y_test, model_name):
    global figure_counter
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted size
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f'Confusion Matrix ({model_name})')
    increment_figure_counter()
    return fig

# Overlayed Density Plot
def plot_density(model, X_test, y_test, model_name):
    global figure_counter
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted size
    sns.kdeplot(
        y_pred_proba[y_test == 0],
        label='Class 0 (Not Injured)',
        fill=True,
        alpha=0.5,
        color='blue',
        ax=ax
    )
    sns.kdeplot(
        y_pred_proba[y_test == 1],
        label='Class 1 (Injured)',
        fill=True,
        alpha=0.5,
        color='orange',
        ax=ax
    )
    ax.set_xlabel('Predicted Probability for Class 1')
    ax.set_ylabel('Density')
    ax.set_title(f'Density Plot ({model_name})')
    ax.legend()
    increment_figure_counter()
    return fig

# Interactive 3D Scatter Plot with Plotly
def plot_3d_predictions(model, X_test, y_test, feature_names, model_name):
    if len(feature_names) < 2:
        print("3D plot requires at least two features.")
        return None

    y_pred = model.predict_proba(X_test)[:, 1]
    feature_1 = X_test[:, 0] if isinstance(X_test, np.ndarray) else X_test.iloc[:, 0]
    feature_2 = X_test[:, 1] if isinstance(X_test, np.ndarray) else X_test.iloc[:, 1]

    fig = px.scatter_3d(
        x=feature_1,
        y=feature_2,
        z=y_pred,
        color=y_test,
        title=f'3D Scatter Plot ({model_name})',
        labels={'x': feature_names[0], 'y': feature_names[1], 'z': 'Predicted Probability'},
        color_continuous_scale='coolwarm',
        opacity=0.8
    )
    fig.update_layout(height=600, width=600)  # Adjusted size for Streamlit
    return fig
