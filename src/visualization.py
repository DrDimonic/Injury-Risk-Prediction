import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    classification_report
)
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.metrics import classification_report

# Adjust global figure size for smaller visuals
SMALL_FIGSIZE = (6, 4)

# Plot the feature importances of the model.
def plot_feature_importances(model, feature_names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    ax.barh(sorted_feature_names, importances[indices], color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Feature Importances ({model_name})")
    plt.tight_layout()
    return fig

# Plot a correlation heatmap
def plot_correlation_heatmap(data):
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)

    heatmap = sns.heatmap(
        data.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 3},  # Font size for annotations
        cbar_kws={"format": "%.2f"},  # Format for color bar numbers
        ax=ax,
    )

    # Adjust font size for axis tick labels
    ax.tick_params(axis="x", labelsize=5)  # X-axis tick labels
    ax.tick_params(axis="y", labelsize=5)  # Y-axis tick labels

    # Adjust font size for color bar ticks
    heatmap.figure.axes[-1].tick_params(labelsize=5)  # Color bar tick labels

    # Set title font size
    ax.set_title("Correlation Heatmap", fontsize=10)

    plt.tight_layout()
    return fig

# 3D scatter plot for predictions
def plot_3d_predictions(model, X_test, y_test, feature_names, model_name):
    # Ensure valid features for plotting
    if len(feature_names) < 2:
        print("3D plot requires at least two features.")
        return None

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)

    # Extract first two features
    feature_1 = X_test[:, 0] if isinstance(X_test, np.ndarray) else X_test.iloc[:, 0]
    feature_2 = X_test[:, 1] if isinstance(X_test, np.ndarray) else X_test.iloc[:, 1]

    fig = go.Figure(
        data=go.Scatter3d(
            x=feature_1,
            y=feature_2,
            z=y_pred,
            mode="markers",
            marker=dict(size=5, color=y_test, colorscale="Viridis", opacity=0.8),
        )
    )
    fig.update_layout(
        title=f"3D Scatter Plot: Predictions ({model_name})",
        scene=dict(
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            zaxis_title="Predicted Probabilities",
        ),
        width=600,
        height=400,  # Smaller dimensions
    )
    return fig

# Plot confusion matrix
def plot_confusion_matrix(model, X_test, y_test, model_name):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix ({model_name})", fontsize=10)
    plt.tight_layout()
    return fig

# Plot density plot of predicted probabilities
def plot_density(model, X_test, y_test, model_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    sns.kdeplot(
        y_pred_proba[y_test == 0],
        label="Class 0 (Not Injured)",
        fill=True,
        alpha=0.5,
        color="blue",
        ax=ax,
    )
    sns.kdeplot(
        y_pred_proba[y_test == 1],
        label="Class 1 (Injured)",
        fill=True,
        alpha=0.5,
        color="orange",
        ax=ax,
    )
    ax.set_xlabel("Predicted Probability for Class 1")
    ax.set_ylabel("Density")
    ax.set_title(f"Density Plot of Predicted Probabilities ({model_name})", fontsize=10)
    ax.legend()
    plt.tight_layout()
    return fig

# Plot precision-recall curve
def plot_precision_recall_curve(model, X_test, y_test, model_name):
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)

    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    ax.plot(recall, precision, label=f"Avg Precision = {avg_precision:.2f}", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve ({model_name})", fontsize=10)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig

# Plot ROC curve
def plot_roc_curve(model, X_test, y_test, model_name):
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)  # Diagonal line for reference
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({model_name})", fontsize=10)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig

def plot_classification_report(model, X_test, y_test, model_name, scaler=None):
    # Apply scaling if required
    if scaler:
        X_test = scaler.transform(X_test)

    # Generate predictions and classification report
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, target_names=["Class 0", "Class 1"])

    # Create a figure to display the report
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")  # Hide the axes

    # Add the title above the report
    ax.text(0.5, 1.1, f"Classification Report\n({model_name})", fontsize=12, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.5, report, fontsize=10, ha="center", va="center", family="monospace", transform=ax.transAxes)

    plt.tight_layout()
    return fig

