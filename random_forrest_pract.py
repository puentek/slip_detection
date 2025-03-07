# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to load and merge multiple CSV files
def load_and_merge_data(file_paths):
    """Loads sensor data from multiple CSV files and merges them into one DataFrame."""
    combined_df = pd.DataFrame()  # Initialize empty DataFrame

    for file_path in file_paths:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)

        # Ensure correct column names
        expected_columns = [f"Sensor{i+1}" for i in range(8)]
        if list(df.columns) != expected_columns:
            print("Warning: CSV columns do not match expected format. Renaming columns...")
            df.columns = expected_columns  # Assign default names

        # Add a source column (optional, to track where data came from)
        df["Source"] = file_path

        # Automatically assign Slip labels based on pressure fluctuation rule
        df["Slip"] = np.where(df.iloc[:, :7].std(axis=1) > df.iloc[:, 7].std(), 1, 0)

        # Append to the combined dataset
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    print("\nAll data files merged successfully!")
    return combined_df

# Function to visualize sensor data trends
def plot_sensor_trends(df):
    """Plots the sensor data trends over time."""
    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.plot(df.index, df[f"Sensor{i+1}"], label=f"Sensor {i+1}", alpha=0.7)
    
    plt.xlabel("Time Step")
    plt.ylabel("Pressure Reading")
    plt.title("Sensor Pressure Trends Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to visualize slip vs no-slip distribution
def plot_slip_distribution(df):
    """Plots the distribution of slip vs no-slip cases."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["Slip"], palette=["red", "blue"])
    plt.xticks([0, 1], ["No Slip", "Slip"])
    plt.xlabel("Slip Condition")
    plt.ylabel("Count")
    plt.title("Distribution of Slip vs No Slip")
    plt.grid(axis="y")
    plt.show()

# Function to train, evaluate, and visualize feature importance
def train_and_evaluate(df):
    """Trains a Random Forest classifier and evaluates performance."""
    X = df.drop(columns=["Slip", "Source"])  # Features (exclude slip label & source column)
    y = df["Slip"]  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n===== Final Model Evaluation =====")
    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:\n", report)

    # Feature Importance Plot
    feature_importances = rf.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_importances, y=[f"Sensor{i+1}" for i in range(8)], palette="viridis")
    plt.xlabel("Importance Score")
    plt.ylabel("Sensor")
    plt.title("Feature Importance (Which Sensors Matter Most for Slip Detection)")
    plt.grid(True)
    plt.show()

# Main function to load data, merge, train model, and visualize
def main(file_paths):
    merged_df = load_and_merge_data(file_paths)

    # Visualization
    plot_sensor_trends(merged_df)
    plot_slip_distribution(merged_df)

    # Train and evaluate model
    train_and_evaluate(merged_df)


# List of 10 CSV files (update these with actual file names)
csv_files = [
    'trial1.csv', 'trial2.csv', 'trial3.csv', 'trial4.csv', 'trial5.csv',
    'trial6.csv', 'trial7.csv', 'trial8.csv', 'trial9.csv', 'trial10.csv'
]

# Uncomment and update with actual file paths before running
main(csv_files)


