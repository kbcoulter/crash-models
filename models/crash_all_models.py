"""
crash_models.py: All Crash Models Condensed Into a Single Script
Updated to match personal organization and style for GitHub | KC | 02/2025 
Visit README.md for more information
"""

##### IMPORTS #####
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pickle
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

##### ACCESS DATA #####
with open('data_variables.pkl', 'rb') as file:
    X_train_temp_small, y_train_temp_small, X_temp_small, y_temp_small, X_val_small, y_val_small, X_test_small, y_test_small,X_train_scale_small,X_val_scale_small,X_test_scale_small = pickle.load(file)

########## Logistic Regression Model for Classification ##########

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

########## Decision Tree Model for Classification ##########

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

########## Random Forest Model for Classification ##########

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

########## Neural Network for Classification ##########
##### MODEL #####
def create_neural_network(): # Function to create a neural network in Keras using Sequential API
    # 2 hidden layers with 64 and 32 neurons, respectively, and ReLU activation functions
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scale_small.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # With adam optimizer for iteration of 10 seeds
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# HOLDERS FOR ACCURACY
accuracy_val = []
accuracy_train = []
accuracy_test = []
seeds = 10 # 10 seeds for iteration

# LOOP THROUGH THE SEEDS
for seed in range(seeds):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # CREATE AND FIT THE NERUAL NETWORK
    neural_network = create_neural_network()
    neural_network.fit(X_train_scale_small, y_train_temp_small, epochs=10, batch_size=32, verbose=0)
    
    # VALIDATION
    y_val_pred_nn = (neural_network.predict(X_val_scale_small) > 0.5).astype("int32") #AS INT
    accuracy_val.append(accuracy_score(y_val_small, y_val_pred_nn))

    # TRAIN
    y_train_pred_nn = (neural_network.predict(X_train_scale_small) > 0.5).astype("int32")
    accuracy_train.append(accuracy_score(y_train_temp_small, y_train_pred_nn))

    # TEST
    y_test_pred_nn = (neural_network.predict(X_test_scale_small) > 0.5).astype("int32")
    accuracy_test.append(accuracy_score(y_test_small, y_test_pred_nn))

# AVERAGES
average_accuracy_val = np.mean(accuracy_val)
average_accuracy_train = np.mean(accuracy_train)
average_accuracy_test = np.mean(accuracy_test)

# PRINT
print("NN Train Accuracy:", average_accuracy_train)
print("NN Val Accuracy:", average_accuracy_val)
print("NN Test Accuracy:", average_accuracy_test)

####################### LINEAR KERNEL #######################

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

#################### POLYNOMIAL KERNEL ####################

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
