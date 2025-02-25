"""
log_reg_model.py: Logistic Regression Model for Classification
    Updated to match personal organization and style for GitHub | KC | 02/2025 
Visit README.md for more information
ORIGINAL DATA SOURCE: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
"""

##### IMPORTS #####
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings

##### ACCESS DATA #####
with open('data_variables.pkl', 'rb') as file:
    X_train_temp_small, y_train_temp_small, X_temp_small, y_temp_small, X_val_small, y_val_small, X_test_small, y_test_small,X_train_scale_small,X_val_scale_small,X_test_scale_small = pickle.load(file)

##### MODEL #####
log_reg_model = LogisticRegressionCV(max_iter = 10000, cv = 7, random_state = 12) # Initialize logreg model
log_reg_model.fit(X_train_scale_small, y_train_temp_small) # Fit Model

# LASSO SHOWS NO IMMEDIATE IMPROVEMENT, NOT INCLUDED IN FINAL MODEL
# log_reg_model =LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=11)

# TRAINING
y_train_pred = log_reg_model.predict(X_train_scale_small)
train_acc = accuracy_score(y_train_temp_small, y_train_pred)
print("LR Training Accuracy:", train_acc)
print("LR Training Report:")
print(classification_report(y_train_temp_small, y_train_pred))

# VALIDATION
y_val_pred = log_reg_model.predict(X_val_scale_small)
val_acc = accuracy_score(y_val_small, y_val_pred)
print("LR Validation Accuracy:", val_acc)
print("LR Validation Report:")
print(classification_report(y_val_small, y_val_pred))

# TESTING
y_test_pred = log_reg_model.predict(X_test_scale_small)
test_acc = accuracy_score(y_test_small, y_test_pred)
print("LR Test Accuracy:", test_acc)
print("LR Test Report:")
print(classification_report(y_test_small, y_test_pred))