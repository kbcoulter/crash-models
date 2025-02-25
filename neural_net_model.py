"""
neural_net_model.py: Neural Network for Classification
    Updated to match personal organization and style for GitHub | KC | 02/2025 
Visit README.md for more information
ORIGINAL DATA SOURCE: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
"""

##### IMPORTS #####
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings

##### ACCESS DATA #####
with open('data_variables.pkl', 'rb') as file:
    X_train_temp_small, y_train_temp_small, X_temp_small, y_temp_small, X_val_small, y_val_small, X_test_small, y_test_small,X_train_scale_small,X_val_scale_small,X_test_scale_small = pickle.load(file)

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