import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV file
df = pd.read_csv("trial8_with_time.csv")  # Replace with your actual file name


# If there's no time column, create one assuming a 10 Hz sampling rate
if "Time" not in df.columns:
    df["Time"] = df.index * 0.1  # Assuming a 10 Hz sampling rate

# Apply slip detection rule:
df["Slip"] = ((df.iloc[:, :7] > 500).any(axis=1)) | (df["Sensor8"] < 1000)
df["Slip"] = df["Slip"].astype(int)  # Convert boolean to 0/1

# Identify slip event times
slip_times = df[df["Slip"] == 1]["Time"]
slip_intervals = []
current_start = None

# Identify continuous slip intervals
for i in range(len(df)):
    if df["Slip"].iloc[i] == 1:
        if current_start is None:
            current_start = df["Time"].iloc[i]  # Start of slip event
    else:
        if current_start is not None:
            slip_intervals.append((current_start, df["Time"].iloc[i]))  # End of slip event
            current_start = None

# If the last slip event extends to the end of the data
if current_start is not None:
    slip_intervals.append((current_start, df["Time"].iloc[-1]))

# Define number of subplots
num_sensors = 8  # Number of sensors
fig, axes = plt.subplots(num_sensors + 1, 1, figsize=(12, 12), sharex=True)

# Plot each sensor's values over time
for i in range(num_sensors):
    axes[i].plot(df["Time"], df[f"Sensor{i+1}"], label=f"Sensor {i+1}", alpha=0.7)
    axes[i].set_ylabel(f"Sensor {i+1}")
    axes[i].legend()
    axes[i].grid(True)

    # Add shaded slip intervals
    for start, end in slip_intervals:
        axes[i].axvspan(start, end, color="red", alpha=0.2)

# Plot slip status in the last subplot
axes[num_sensors].plot(df["Time"], df["Slip"], marker="o", linestyle="-", color="red", alpha=0.7)
axes[num_sensors].set_ylabel("Slip Status (0=No Slip, 1=Slip)")
axes[num_sensors].set_xlabel("Time (s)")
axes[num_sensors].set_yticks([0, 1])  # Show only 0 and 1
axes[num_sensors].legend(["Slip"])
axes[num_sensors].grid(True)

# Add shaded slip intervals to the slip plot
for start, end in slip_intervals:
    axes[num_sensors].axvspan(start, end, color="red", alpha=0.2)

# Set title
fig.suptitle("Sensor Readings and Slip Detection Over Time", fontsize=14)

# Save the plot as PNG and PDF
plt.savefig("slip_detection_plot_trial8.png", dpi=300, bbox_inches="tight")
plt.savefig("slip_detection_plot_trial8.pdf", dpi=300, bbox_inches="tight")

plt.show()


