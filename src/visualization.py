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

    plt.figure(1) 
    plt.barh(sorted_feature_names, importances[indices])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

# Plot actual vs predicted values.
def plot_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    plt.figure(2)  
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

# Plot the confusion matrix.
def plot_confusion_matrix(model, X_test, y_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.figure(3) 
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# Plot histogram of actual vs predicted values.
def plot_actual_vs_predicted_histogram(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    plt.figure(4)  
    plt.hist(y_test, bins=20, alpha=0.5, label='Actual')
    plt.hist(predictions, bins=20, alpha=0.5, label='Predicted')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()

