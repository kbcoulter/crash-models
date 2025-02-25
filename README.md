# crash_models
This project uses machine learning to predict the impact of car accidents on nearby traffic delays. The goal is to classify whether an accident will cause a severe delay or not, helping to improve traffic management and response strategies.

This repository contains a small project for DSCI/CS 372: Machine Learning for Data Science at the University of Oregon. Some formatting changes have been made to better align with my preferences.

# Data
The [US Accidents (2016 - 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) dataset spans 49 U.S. states and contains accident data collected from various APIs between February 2016 and March 2023. Sources include national and state departments of transportation, law enforcement agencies, traffic cameras, and more. Initially, the dataset comprised around 7.7 million accidents across 46 variables.

# What’s Included:
- [crash_prep.py](https://github.com/kbcoulter/crash-models/blob/main/crash_prep.py): A python script for data preprocessing
- [models](https://github.com/kbcoulter/crash-models/blob/main/models/): A folder containing the following models:
    - [Feedforward Neural Network](https://github.com/kbcoulter/crash-models/blob/main/models/nerual_net_model.py)
    - [Random Forest](https://github.com/kbcoulter/crash-models/blob/main/models/rand_forest_model.py)
    - [Decision Tree](https://github.com/kbcoulter/crash-models/blob/main/models/dec_tree_model.py)
    - [Linear SVM](https://github.com/kbcoulter/crash-models/blob/main/models/svm_lin_model.py)
    - [Polynomial SVM](https://github.com/kbcoulter/crash-models/blob/main/models/svm_poly_model.py)
    - [Logistic Regression Model](https://github.com/kbcoulter/crash-models/blob/main/models/log_reg_model.py)
    - [All Models](https://github.com/kbcoulter/crash-models/blob/main/models/crash_all_models.py) (All Models in Single Script)
    - [All Models .ipynb](https://github.com/kbcoulter/crash-models/blob/main/models/crash_all_models.py) (All Models in Jupyter Notebook)

# Model Performance 
*Accuracy values reflect results based on randomly selected random seeds.*


 | **Model**                          | **Training Accuracy** | **Validation Accuracy** | **Test Accuracy** |
|------------------------------------|-----------------------|-------------------------|-------------------|
| Neural Network                     | 0.8546                | 0.8609                  | 0.8632            |
| Random Forest                      | 0.9988                | 0.8558                  | 0.8566            |
| Decision Tree                      | 0.9988                | 0.7558                  | 0.7639            |
| Linear SVM (Kernel 1)               | 0.8545                | 0.8605                  | 0.8636            |
| Polynomial SVM (Kernel 2)           | 0.8556                | 0.8597                  | 0.8625            |
| Logistic Regression                 | 0.8545                | 0.8605                  | 0.8636            |

# Evaluation and Insights
Logistic Regression, Linear SVM, Polynomial SVM, and Neural Network demonstrate consistent accuracy, suggesting stable generalization and strong classification performance.

Unfortunately, the Decision Tree and Random Forest models exhibit signs of overfitting, as shown by high training accuracy paired with lower validation and test accuracy. Without further refinement, it may be best to avoid these models for this classification task due to their clear overfitting tendencies.

The choice of the best model(s) between the Logistic Regression, Linear SVM, Polynomial SVM, and Neural Network models ultimately depends on interpretability, desired implementation, resources, and other constraints.

# Maintenance & Feedback:
Feel free to reach out if you run into any issues, have suggestions, or believe this repo needs updates. I’m happy to maintain this repo if needed/ desired. Thanks!
