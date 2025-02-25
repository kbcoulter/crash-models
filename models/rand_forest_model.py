"""
rand_forest_model.py: Random Forest Model for Classification
    Updated to match personal organization and style for GitHub | KC | 02/2025 
Visit README.md for more information
ORIGINAL DATA SOURCE: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
"""

##### IMPORTS #####
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings

##### ACCESS DATA #####
with open('data_variables.pkl', 'rb') as file:
    X_train_temp_small, y_train_temp_small, X_temp_small, y_temp_small, X_val_small, y_val_small, X_test_small, y_test_small,X_train_scale_small,X_val_scale_small,X_test_scale_small = pickle.load(file)

##### MODEL #####
random_forest_model = RandomForestClassifier(n_estimators=200, random_state=32) # Initialize Random Forest model
random_forest_model.fit(X_train_scale_small, y_train_temp_small) # Fit Model
y_val_pred_rf = random_forest_model.predict(X_val_scale_small)

# TRAINING
y_train_pred_rf = random_forest_model.predict(X_train_scale_small)
rf_train_accuracy = accuracy_score(y_train_temp_small, y_train_pred_rf)
print("RF Training Accuracy:", rf_train_accuracy)
print("RF Training Report:")
print(classification_report(y_train_temp_small, y_train_pred_rf))

# VALIDATION
y_val_pred_rf = random_forest_model.predict(X_val_scale_small)
rf_val_accuracy = accuracy_score(y_val_small, y_val_pred_rf)
print("RF Validation Accuracy:", rf_val_accuracy)
print("RF Validation Report:")
print(classification_report(y_val_small, y_val_pred_rf))

# TESTING
y_test_pred_rf = random_forest_model.predict(X_test_scale_small)
rf_test_accuracy = accuracy_score(y_test_small, y_test_pred_rf)
print("RF Test Accuracy:", rf_test_accuracy)
print("RF Test Report:")
print(classification_report(y_test_small, y_test_pred_rf))
