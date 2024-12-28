import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation
from keras.utils import to_categorical

# Directory containing audio files
audio_dir = r"C:\Users\Toriichi\Desktop\Emotion Detection\Emotion_detection_in_video_calls\wav"

# List all .wav files in the directory
file_paths = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith(".wav")]

# Function to extract MFCC features
def load_audio_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

# Load dataset
data = []  # List to hold features
labels = []  # List to hold labels (e.g., 0 for 'angry', 1 for 'happy', etc.)

for i, file_path in enumerate(file_paths):
    features = load_audio_features(file_path)
    data.append(features)
    labels.append(i % 8)  # Simulating 8 emotions (replace with actual labels)

data = np.array(data)
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=8)
y_test = to_categorical(y_test, num_classes=8)

# Build the model
model = Sequential([
    Conv1D(256, 8, padding='same', activation='relu', input_shape=(40, 1)),
    Conv1D(128, 8, padding='same', activation='relu'),
    Dropout(0.25),
    MaxPooling1D(pool_size=8),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train[..., np.newaxis], y_train, epochs=50, batch_size=32, validation_data=(X_test[..., np.newaxis], y_test))

# Save the model
model.save('emotion_recognition_model.h5')
print("Model saved as emotion_recognition_model.h5")
