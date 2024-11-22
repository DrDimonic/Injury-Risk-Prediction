from src.data_processing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model
from sklearn.model_selection import train_test_split
import joblib

def main():
    # Load and preprocess the data
    data = load_data(r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\data\dataset.csv")
    features, target = preprocess_data(data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model for future use
    joblib.dump(model, r"C:\Users\domin\Data Mining Project\Injury-Risk-Prediction\models\trained_model.pkl")
    print("Model saved as 'trained_model.pkl'")

if __name__ == "__main__":
    main()