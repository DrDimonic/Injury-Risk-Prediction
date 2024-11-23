from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Train Random Forest model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# Train Logistic Regression model
def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logreg = LogisticRegression(solver='lbfgs', max_iter=10000, class_weight='balanced', random_state=42)
    logreg.fit(X_train_scaled, y_train)
    return logreg, scaler

# Evaluate the model
def evaluate_model(model, X_test, y_test, scaler=None):
    if scaler:  # Scale test features for Logistic Regression
        X_test = scaler.transform(X_test)

    predictions = model.predict(X_test)

    # Print evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    return predictions

# Compare Random Forest and Logistic Regression models
def compare_models(X_train, y_train, X_test, y_test):
    print("\n=== Training Random Forest ===")
    rf_model = train_random_forest(X_train, y_train)
    print("\nRandom Forest Evaluation:")
    evaluate_model(rf_model, X_test, y_test)

    print("\n=== Training Logistic Regression ===")
    logreg_model, scaler = train_logistic_regression(X_train, y_train)
    print("\nLogistic Regression Evaluation:")
    evaluate_model(logreg_model, X_test, y_test, scaler)

    # Perform cross-validation for both models
    print("\n=== Cross-validation Scores ===")
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"Random Forest F1 Macro Cross-Validation Scores: {rf_cv_scores}")
    print(f"Mean Random Forest F1 Score: {rf_cv_scores.mean()}")

    logreg_cv_scores = cross_val_score(logreg_model, scaler.transform(X_train), y_train, cv=5, scoring='f1_macro')
    print(f"Logistic Regression F1 Macro Cross-Validation Scores: {logreg_cv_scores}")
    print(f"Mean Logistic Regression F1 Score: {logreg_cv_scores.mean()}")
