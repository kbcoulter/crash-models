"""
svm_poly_model.py: Polynomial Support Vector Machine for Classification
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
poly_model = SVC(kernel='poly', degree=3, random_state=54) # SVM with Polynomial kernel
poly_model.fit(X_train_scale_small, y_train_temp_small)

# TRAINING
poly_y_train_pred = poly_model.predict(X_train_scale_small)
poly_train_accuracy = accuracy_score(y_train_temp_small, poly_y_train_pred)
print("Poly SVM Training Accuracy:", poly_train_accuracy)
print("Poly SVM Training Report:")
print(classification_report(y_train_temp_small, poly_y_train_pred))

# VALIDATION
poly_y_val_pred = poly_model.predict(X_val_scale_small)
poly_val_accuracy = accuracy_score(y_val_small, poly_y_val_pred)
print("Poly SVM Validation Accuracy:", poly_val_accuracy)
print("Poly SVM Validation Report:")
print(classification_report(y_val_small, poly_y_val_pred))

# TESTING
poly_y_test_pred = poly_model.predict(X_test_scale_small)
poly_test_accuracy = accuracy_score(y_test_small, poly_y_test_pred)
print("Poly SVM Test Accuracy:", poly_test_accuracy)
print("Poly SVM Test Report:")
print(classification_report(y_test_small, poly_y_test_pred))