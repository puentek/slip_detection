import pandas as pd

# Load the CSV file
df = pd.read_csv("trial10.csv")  # Replace with your file path

# Estimate the sampling rate (Assume 40 seconds total)
recording_duration = 40  # seconds
num_samples = len(df)  # Total rows
sampling_rate = num_samples / recording_duration  # Samples per second

# Generate timestamps
df["Time (s)"] = df.index / sampling_rate  # Time in seconds

# Save the updated CSV with timestamps
df.to_csv("trial10_with_time.csv", index=False)

print(f"Estimated Sampling Rate: {sampling_rate:.2f} Hz")
print(df.head())  # Show first few rows
