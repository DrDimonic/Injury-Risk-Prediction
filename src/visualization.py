import matplotlib.pyplot as plt

def plot_feature_importances(model, feature_names):
    """Plot feature importance from a trained model."""
    importances = model.feature_importances_
    sorted_indices = importances.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[sorted_indices], align='center')
    plt.yticks(range(len(importances)), feature_names[sorted_indices])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()
