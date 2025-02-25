"""
svm_lin_model.py: Linear Support Vector Machine for Classification
    Updated to match personal organization and style for GitHub | KC | 02/2025 
Visit README.md for more information
ORIGINAL DATA SOURCE: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
"""

##### IMPORTS #####
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.svm import SVC
import warnings

##### ACCESS DATA #####
with open('data_variables.pkl', 'rb') as file:
    X_train_temp_small, y_train_temp_small, X_temp_small, y_temp_small, X_val_small, y_val_small, X_test_small, y_test_small,X_train_scale_small,X_val_scale_small,X_test_scale_small = pickle.load(file)

# CREATE MODEL
linear_model = SVC(kernel='linear', random_state=54) # SVM with Linear kernel
linear_model.fit(X_train_scale_small, y_train_temp_small)

# TRAINING
linear_y_train_pred = linear_model.predict(X_train_scale_small)
linear_train_accuracy = accuracy_score(y_train_temp_small, linear_y_train_pred)
print("Linear SVM Training Accuracy:", linear_train_accuracy)
print("Linear SVM Training Report:")
print(classification_report(y_train_temp_small, linear_y_train_pred))

# VALIDATION
linear_y_val_pred = linear_model.predict(X_val_scale_small)
linear_val_accuracy = accuracy_score(y_val_small, linear_y_val_pred)
print("Linear SVM Validation Accuracy:", linear_val_accuracy)
print("Linear SVM Validation Report:")
print(classification_report(y_val_small, linear_y_val_pred))

# TESTING
linear_y_test_pred = linear_model.predict(X_test_scale_small)
linear_test_accuracy = accuracy_score(y_test_small, linear_y_test_pred)
print("Linear SVM Test Accuracy:", linear_test_accuracy)
print("Linear SVM Test Report:")
print(classification_report(y_test_small, linear_y_test_pred))
