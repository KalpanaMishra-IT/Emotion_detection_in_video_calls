import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

# Define input and output directories (update this to match your environment)
input_dir = r"D:\BRUHHH ITS THE FINAL CODEEEE\Emotion_detection_in_video_calls-main"
output_dir = r"D:\BRUHHH ITS THE FINAL CODEEEE\Emotion_detection_in_video_calls-main"

def main():
    # Ensure the output directory exists
    try:
        if not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    # Define file paths
    x_train_path = os.path.join(input_dir, "X_train.npy")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    scaled_features_path = os.path.join(output_dir, "X_train_scaled.npy")

    # Debug path details
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"X_train.npy path: {x_train_path}")
    print(f"Scaler save path: {scaler_path}")
    print(f"Scaled features save path: {scaled_features_path}")

    # Check if X_train.npy exists
    if not os.path.exists(x_train_path):
        print(f"Error: File '{x_train_path}' does not exist.")
        return

    # Load X_train
    print(f"Loading features from {x_train_path}...")
    X_train = np.load(x_train_path)

    # Validate X_train
    if X_train.size == 0:
        print("Error: X_train is empty.")
        return

    print(f"Original X_train shape: {X_train.shape}")

    # Apply scaling
    try:
        print("Scaling features using StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print("Scaling completed successfully.")

        # Save the scaled features
        np.save(scaled_features_path, X_train_scaled)
        print(f"Scaled features saved at: {scaled_features_path}")

        # Save the scaler object
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at: {scaler_path}")
    except Exception as e:
        print(f"Error during scaling or saving: {e}")

if __name__ == "__main__":
    print("Starting scaling process...")
    main()
