from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Train Random Forest model.
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# Train Logistic Regression model.
def train_logistic_regression(X_train, y_train):
    # Apply Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the model
    logreg = LogisticRegression(solver='lbfgs', max_iter=10000, class_weight='balanced', random_state=42)
    logreg.fit(X_train_scaled, y_train)

    return logreg, scaler

# Evaluate the model using test data.
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    return predictions

# Compare Logistic Regression and Random Forest models.
def compare_models(X_train, y_train, X_test, y_test):
    """Compare the performance of Logistic Regression and Random Forest models."""

    # Train Logistic Regression
    logreg = train_logistic_regression(X_train, y_train)
    print("\nLogistic Regression Evaluation:")
    evaluate_model(logreg, X_test, y_test)

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    print("\nRandom Forest Evaluation:")
    evaluate_model(rf_model, X_test, y_test)
