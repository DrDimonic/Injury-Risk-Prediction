from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from sklearn.model_selection import GridSearchCV

# Train a Random Forest model using balanced data.
def train_model(X_train, y_train):
    smote_enn = SMOTEENN(random_state=42)
    X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)

    # Define Random Forest model
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Define hyperparameters for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Perform grid search for best hyperparameters
    grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)

    best_model = grid_search.best_estimator_
    return best_model

# Evaluate the model on the test data.
def evaluate_model(model, X_test, y_test):
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Model Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    return accuracy
