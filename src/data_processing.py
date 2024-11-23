import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    # Load data from a CSV file.
    try:
        data = pd.read_csv(filepath)
        print("Dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Preprocess the dataset for training
def preprocess_data(data):
    # Separate features and target
    features = data.iloc[:, :-1]  # All columns except the last
    target = data.iloc[:, -1]    # Last column as target

    # Apply Standard Scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print("Preprocessing complete with Standard Scaling.")
    return features_scaled, target, scaler
