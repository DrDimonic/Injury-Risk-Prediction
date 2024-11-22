from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Random Forest model.
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# Logistic Regression model.
def train_logistic_regression(X_train, y_train):
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logreg = LogisticRegression(solver='lbfgs', max_iter=10000, class_weight='balanced', random_state=42)
    logreg.fit(X_train_scaled, y_train)

    return logreg, scaler

# Evaluate the model using test data.
def evaluate_model(model, X_test, y_test, scaler=None):
    if scaler:  
        X_test = scaler.transform(X_test)

    predictions = model.predict(X_test)

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return predictions

# Perform cross-validation on Random Forest.
def cross_validate_model(features, target):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv_scores = cross_val_score(model, features, target, cv=5, scoring='f1_macro')
    print("Cross-validation F1 scores:", cv_scores)
    print("Mean F1 score:", cv_scores.mean())

# Compare Logistic Regression and Random Forest models.
def compare_models(X_train, y_train, X_test, y_test):

    # Logistic Regression
    logreg, scaler = train_logistic_regression(X_train, y_train)
    print("\nLogistic Regression Evaluation:")
    logreg_predictions = evaluate_model(logreg, X_test, y_test, scaler)

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    print("\nRandom Forest Evaluation:")
    rf_predictions = evaluate_model(rf_model, X_test, y_test)

