import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os 

# 1st lets specify the folder path its in 
folder_path ="C:/Users/18328/Desktop/onr_tact/trial_data/"
# now we get all the files we want to use 
file_names = {
    'trial1_with_time.csv', 'trial2_with_time.csv', 'trial3_with_time.csv', 'trial4_with_time.csv', 'trial5_with_time.csv',
    'trial6_with_time.csv', 'trial7_with_time.csv', 'trial8_with_time.csv', 'trial9_with_time.csv', 'trial10_with_time.csv'
};

# combine the paths 
file_paths = [os.path.join(folder_path,file) for file in file_names]

# # Load the csv files with the file names 
# for file in file_paths:
#     df = pd.read_csv(file)
#     print(f"Loaded {file} successfully!")

#  here i am defining that the 1st 8 files are training and the last 2 are testing  
train_files = 8
test_files = len(file_paths)- train_files

#Initialize trainign and testing data containers 
X_train, y_train = [], []
X_test, y_test, time_test = [], [], []

#  process the data from the csv files 
for i, file in enumerate(file_paths):
    # load file function 
    df = pd.read_csv(file)

    # check the strcuture of the csv files 
    required_columns = [f"Sensor{i}" for i in range(1,9)] +["Time (s)"] 
    if not all (col in df.columns for col in required_columns):
        raise ValueError(f"CSV file {file} is missing required columns: {required_columns}")
    
    # Extract time, sensor readings, and define slip labels 
    time = df["Time (s)"] # Extract time for plotting (not training)
    # Extract sensor 1-8 
    sensors = df.iloc[:,1:9] 
    slip_labels = ((df["Sensor8"]< 1000) | (df.iloc[:,1:8] > 500).any(axis=1)).astype(int)

    # split into trainign and testing sets 
    if i < train_files:
        X_train.append(sensors)
        y_train.append(slip_labels)
    else:
        X_test.append(sensors)
        y_test.append(slip_labels)
        time_test.append(time)

# Convert lists to numpy arrays for model training 
X_train = pd.concat(X_train).values
y_train = pd.concat(y_train).values
X_test = pd.concat(X_test).values
y_test = pd.concat(y_test).values
time_test = pd.concat(time_test).values

# train the random forest classifier now
print(f"Training rand forest on {train_files} datasets... ")
rf_model = RandomForestClassifier(n_estimators=5, random_state=42)
rf_model.fit(X_train,y_train)

#  test tje model on the unseen data
print(f"Testning model on {test_files} datasets...")
y_pred = rf_model.predict(X_test)

#calculate the accuracy 
accuracy = accuracy_score(y_test,y_pred) 
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(12, 6))

# Shade actual slip regions (y_test == 1) in light blue
plt.fill_between(time_test, 0, 1, where=(y_test == 1), color='lightblue', alpha=0.5, label='Actual Slip')

# Plot actual slip values (Blue circles)
plt.plot(time_test, y_test, 'bo-', markersize=4, label='Actual Slip')

# Plot predicted slip values (Red dots)
plt.plot(time_test, y_pred, 'r.-', markersize=10, label='Predicted Slip')

plt.xlabel('Time (s)')
plt.ylabel('Slip Status (0 = No Slip, 1 = Slip)')
plt.title('Actual vs Predicted Slip Detection Over Time')
plt.legend()
plt.grid(True)

plt.show()