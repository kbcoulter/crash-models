"""
dec_tree_model.py: Decision Tree Model for Classification
    Updated to match personal organization and style for GitHub | KC | 02/2025 
Visit README.md for more information
ORIGINAL DATA SOURCE: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
"""

##### IMPORTS #####
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.tree import DecisionTreeClassifier
import warnings

##### ACCESS DATA #####
with open('data_variables.pkl', 'rb') as file:
    X_train_temp_small, y_train_temp_small, X_temp_small, y_temp_small, X_val_small, y_val_small, X_test_small, y_test_small,X_train_scale_small,X_val_scale_small,X_test_scale_small = pickle.load(file)

##### MODEL #####
tree_model=DecisionTreeClassifier(random_state=23) # Initialize Decision Tree model
tree_model.fit(X_train_scale_small, y_train_temp_small) # Fit Model

# TRAINING
tree_y_train_pred = tree_model.predict(X_train_scale_small)
tree_train_accuracy = accuracy_score(y_train_temp_small, tree_y_train_pred)
print("DT Training Accuracy:", tree_train_accuracy)
print("DT Training Report:")
print(classification_report(y_train_temp_small, tree_y_train_pred))

# VALIDATION
tree_y_val_pred = tree_model.predict(X_val_scale_small)
tree_val_accuracy = accuracy_score(y_val_small, tree_y_val_pred)
print("DT Validation Accuracy:", tree_val_accuracy)
print("DT Validation Report:")
print(classification_report(y_val_small, tree_y_val_pred))

# TESTING
tree_y_test_pred = tree_model.predict(X_test_scale_small)
tree_test_accuracy = accuracy_score(y_test_small, tree_y_test_pred)
print("DT Test Accuracy:", tree_test_accuracy)
print("DT Test Report:")
print(classification_report(y_test_small, tree_y_test_pred))