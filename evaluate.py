from src.data_processing import load_data, preprocess_data
from src.model_training import evaluate_model
import joblib

def main():
    # Load the trained model
    model = joblib.load('trained_model.pkl')
    print("Model loaded successfully!")

    # Load and preprocess the data
    data = load_data('data/dataset.csv')
    features, target = preprocess_data(data)

    # Split test set if needed (or evaluate entire dataset)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
