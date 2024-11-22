from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# Train a Random Forest model and return the model, X_test and y_test.
def train_model(features, target):
    """Train a Random Forest model and return the model, X_test, and y_test."""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test


# Evaluate the model on test data and print results.
def evaluate_model(model, X_test, y_test):
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Model Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    return accuracy
