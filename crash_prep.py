"""
crash_prep.py: Prepares crash data to be used in subsequent scripts and models
Updated to match personal organization and style for GitHub | KC | 02/2025 
Visit README.md for more information

WARNING: Some IOS devices may have issues with the encoding of the data.
    ! All Users: Data is large, so script cuts down data to 1% of original data for runtime purposes -> User Settings Line 24
    ! All Users: If you encounter an error, try changing the encoding to 'utf-8' in the pd.read_csv() function. Line 29

ORIGINAL DATA SOURCE:
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
"""

##### IMPORTS #####
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

##### SETTINGS #####
np.random.seed(21) # SET RANDOM SEED FOR REPRODUCIBILITY
percentage_to_keep = 0.01 # Keeping 50_000 crashes total (1% of original data)

##### DATA ORGANIZATION & PREPROCESSING #####
### DF PREP ###
df = pd.read_csv("US_Accidents_March23.csv", encoding='latin1')

crash=df.drop(columns = ["Source",'Civil_Twilight', 'Nautical_Twilight', # Drop Unnecessary Columns
       'Astronomical_Twilight','Timezone','Start_Lat',
       'Start_Lng', 'End_Lat', 'End_Lng',"Distance(mi)", 
        "Description", "Airport_Code", "Weather_Timestamp",
        "Country", 'County', "Wind_Chill(F)"])

crash['End_Time'] = pd.to_datetime(crash['End_Time']) # DATETIME CONVERSION
crash['Start_Time'] = pd.to_datetime(crash['Start_Time'])

severity_mapping = {1: False, 2: False, 3: True, 4: True} #SEVERITY MAPPING -> 1, 2 = False, 3, 4 = True
crash['Severity'] = crash['Severity'].map(severity_mapping)
crash.rename(columns = {"Severity":"Severe_Delay"}, inplace = True)

# The location characteristics cannot be combined into a single column, Code Cut

crash = crash.dropna() # Drop Missing

crash['Start_Hour'] = crash['Start_Time'].dt.hour # Start Hour
crash['End_Hour'] = crash['End_Time'].dt.hour # End Hour
crash.drop(columns=['Start_Time', 'End_Time'], inplace=True)

boolean_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', # Bool Cols
                'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
                'Traffic_Signal', 'Turning_Loop']

crash[boolean_cols] = crash[boolean_cols].astype(int) # Convert Bool Cols to Int

numerical_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', # Num Cols
                  'Wind_Speed(mph)', 'Precipitation(in)']

crash[numerical_cols] = crash[numerical_cols].astype(int) # Ensure Num Cols are Int

### TRAINING, TEMPORARY, VALIDATION, TEST SPLIT ###

X = crash[['Start_Hour', 'End_Hour', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', # Predictor Variables
       'Visibility(mi)', 'Wind_Speed(mph)','Precipitation(in)', 'Amenity', 'Bump', 'Crossing',
       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']]

y = crash['Severe_Delay'] # Target Variable

X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.2, random_state=42) # Training 80% and Temporary 20% Split

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state = 42) # Temporary into validation 50% and test 50% Split (10% of original data)

### CUTDOWN DATA FROM 5_361_635 CRASHES (OPTIONAL) ###
### (To keep runtime within project bounds) ###

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train_temp)
X_val_scale = scaler.transform(X_val)
X_test_scale = scaler.transform(X_test)

keep_train = int(len(X_train_temp) * percentage_to_keep) # Runtime Data Cut
keep_temp = int(len(X_temp) * percentage_to_keep)
keep_val = int(len(X_val) * percentage_to_keep)
keep_test = int(len(X_test) * percentage_to_keep)

### MORE ORGANIZATION ###
random_train = np.random.choice(len(X_train_temp), keep_train, replace=False)
random_temp = np.random.choice(len(X_temp), keep_temp, replace=False)
random_val = np.random.choice(len(X_val), keep_val, replace=False)
random_test = np.random.choice(len(X_test), keep_test, replace=False)

X_train_temp_small = X_train_temp.iloc[random_train]
y_train_temp_small = y_train_temp.iloc[random_train]
X_temp_small = X_temp.iloc[random_temp]
y_temp_small = y_temp.iloc[random_temp]
X_val_small = X_val.iloc[random_val]
y_val_small = y_val.iloc[random_val]
X_test_small = X_test.iloc[random_test]
y_test_small = y_test.iloc[random_test]

X_train_scale_small = scaler.fit_transform(X_train_temp_small)
X_val_scale_small = scaler.transform(X_val_small)
X_test_scale_small = scaler.transform(X_test_small)

""" 
n_components = 10
pca = PCA(n_components=n_components)
pca.fit(X_train_scale_small)
X_train_pca = pca.transform(X_train_scale_small)
X_val_pca = pca.transform(X_val_scale_small) 
# X_test_pca = pca.transform(X_test_scale_small)
""" # PCA Code -> CUT (Uncomment and Edit to Use)


##### PICKLE DUMP (PICKLING?) DATA #####
with open('data_variables.pkl', 'wb') as file: # Dump Data to be used in other files (models)
    pickle.dump((X_train_temp_small, y_train_temp_small, X_temp_small, y_temp_small, X_val_small, y_val_small,
                 X_test_small, y_test_small,X_train_scale_small,X_val_scale_small,X_test_scale_small), file)