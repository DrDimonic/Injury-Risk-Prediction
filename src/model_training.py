from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Train a Random Forest model using balanced data.
def train_model(X_train_balanced, y_train_balanced):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    return model

# Evaluate the model on the test data.
def evaluate_model(model, X_test, y_test):
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Model Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    return accuracy
