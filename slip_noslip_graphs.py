# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the CSV file
# df = pd.read_csv("trial1_with_time.csv")  # Replace with your file name

# # If there's no time column, create one assuming a 10 Hz sampling rate
# if "Time" not in df.columns:
#     df["Time"] = df.index * 0.1  # Each row is 0.1s apart

# # Apply slip detection rule:
# df["Slip"] = ((df.iloc[:, :7] > 500).any(axis=1)) | (df["Sensor8"] < 1000)
# df["Slip"] = df["Slip"].astype(int)  # Convert boolean to 0/1

# # Plot slip status over time
# plt.figure(figsize=(12, 6))
# plt.plot(df["Time"], df["Slip"], marker="o", linestyle="-", color="red", alpha=0.7)

# plt.xlabel("Time (s)")
# plt.ylabel("Slip Status (0 = No Slip, 1 = Slip)")
# plt.title("Slip Detection Over Time")
# plt.yticks([0, 1], ["No Slip", "Slip"])  # Set Y-axis labels
# plt.grid(True)
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("trial1_with_time.csv")  # Replace with your actual file name

# If there's no time column, create one assuming a 10 Hz sampling rate
if "Time" not in df.columns:
    df["Time"] = df.index * 0.1  # Assuming a 10 Hz sampling rate

# Apply slip detection rule:
df["Slip"] = ((df.iloc[:, :7] > 500).any(axis=1)) | (df["Sensor8"] < 1000)
df["Slip"] = df["Slip"].astype(int)  # Convert boolean to 0/1

# Identify slip event times
slip_times = df[df["Slip"] == 1]["Time"]

# Define number of subplots
num_sensors = 8  # Number of sensors
fig, axes = plt.subplots(num_sensors + 1, 1, figsize=(12, 12), sharex=True)

# Plot each sensor's values over time
for i in range(num_sensors):
    axes[i].plot(df["Time"], df[f"Sensor{i+1}"], label=f"Sensor {i+1}", alpha=0.7)
    axes[i].set_ylabel(f"Sensor {i+1}")
    axes[i].legend()
    axes[i].grid(True)

    # Add slip event markers
    for slip_time in slip_times:
        axes[i].axvline(x=slip_time, color="red", linestyle="dashed", alpha=0.6)

# Plot slip status in the last subplot
axes[num_sensors].plot(df["Time"], df["Slip"], marker="o", linestyle="-", color="red", alpha=0.7)
axes[num_sensors].set_ylabel("Slip Status (0=No Slip, 1=Slip)")
axes[num_sensors].set_xlabel("Time (s)")
axes[num_sensors].set_yticks([0, 1])  # Show only 0 and 1
axes[num_sensors].legend(["Slip"])
axes[num_sensors].grid(True)

# Add slip event vertical lines and text annotations
for slip_time in slip_times:
    axes[num_sensors].axvline(x=slip_time, color="red", linestyle="dashed", alpha=0.6)
    axes[num_sensors].text(slip_time, 1.1, "Slip", color="red", fontsize=10, rotation=90, verticalalignment="bottom")

# Set title
fig.suptitle("Sensor Readings and Slip Detection Over Time", fontsize=14)

plt.show()

