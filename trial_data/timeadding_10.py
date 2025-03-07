import pandas as pd

# Load the CSV file
df = pd.read_csv("trial1.csv")  # Replace with your actual file name

# Define the sampling rate
sampling_rate = 10  # Hz
time_interval = 1 / sampling_rate  # Time step per row (0.1 sec)

# Generate timestamps
df["Time (s)"] = df.index * time_interval  # Multiply row index by time step

# Save the updated CSV with timestamps
df.to_csv("trial1_data_with_time.csv", index=False)

print(df.head())  # Show first few rows
