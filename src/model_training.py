from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Random Forest model.
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model


# Evaluate the model using test data.
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    
# Perform cross-validation on Random Forest.
def cross_validate_model(features, target):
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv_scores = cross_val_score(model, features, target, cv=5, scoring='f1_macro')
    print("Cross-validation F1 scores:", cv_scores)
    print("Mean F1 score:", cv_scores.mean())

    
# Compare Logistic Regression and Random Forest models.
def compare_models(X_train, y_train, X_test, y_test):
    
    # Logistic Regression 
    logreg = LogisticRegression(class_weight='balanced', max_iter=5000, solver = 'saga', random_state=42)
    logreg.fit(X_train, y_train)
    logreg_predictions = logreg.predict(X_test)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, logreg_predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, logreg_predictions))

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))
