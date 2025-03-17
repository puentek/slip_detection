import pandas as pd
import sys
import os 

# Load the CSV file
df = pd.read_csv("C:/Users/18328/Desktop/onr_tact/trial_data/trial10_with_time.csv")

# Select columns 1-7 (Index 0-7 in Python)
df.iloc[:, 0:8] = df.iloc[:, 0:8] / df.iloc[:, 0:8].max()

# Save to a specific folder (modify the path as needed)
output_path = "C:/Users/18328/Desktop/onr_tact/converted_data_for_slip/normalized_file_trial10.csv"  # Windows path

df.to_csv(output_path, index=False)  # Save without the index column

print(f"File saved successfully at: {output_path}")
