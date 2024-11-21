from src.data_processing import load_data, preprocess_data
from src.model_training import train_model

def main():
    # Load and preprocess the data
    data = load_data('data/dataset.csv')
    features, target = preprocess_data(data)

    # Train the model
    model, X_test, y_test = train_model(features, target)

    # Save the model (optional)
    import joblib
    joblib.dump(model, 'trained_model.pkl')
    print("Model saved as 'trained_model.pkl'")

if __name__ == "__main__":
    main()
