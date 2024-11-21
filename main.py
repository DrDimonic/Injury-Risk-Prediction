from src.data_processing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model
from src.visualization import plot_feature_importances

def main():
    # Load and preprocess the data
    data = load_data('data/raw/athlete_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Visualize feature importance
    feature_names = data.columns.drop('injury_risk')
    plot_feature_importances(model, feature_names)

if __name__ == "__main__":
    main()
