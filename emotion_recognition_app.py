import streamlit as st
import librosa
import numpy as np
import pickle
from keras.models import load_model

# Load the saved model and scaler
MODEL_PATH = 'emotion_recognition_model.h5'
SCALER_PATH = 'scaler.pkl'

model = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Extract features function
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mfccs = np.mean(mfccs.T, axis=0)
    chroma = np.mean(chroma.T, axis=0)
    mel = np.mean(mel.T, axis=0)
    return np.hstack([mfccs, chroma, mel])

# Predict emotion
def predict_emotion(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    features = extract_features(y, sr)
    scaled_features = scaler.transform([features])
    scaled_features = np.expand_dims(scaled_features, axis=2)
    prediction = model.predict(scaled_features)
    emotions = ['Angry', 'Calm', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    return emotions[np.argmax(prediction)]

# Streamlit App
def main():
    st.title("🎙 Emotion Recognition from Audio")
    st.write("Upload or record an audio file, and the app will predict the emotion!")

    option = st.radio("Choose an option:", ["Upload an audio file", "Record audio"])

    if option == "Upload an audio file":
        uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])
        if uploaded_file is not None:
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Analyzing..."):
                emotion = predict_emotion("temp_audio.wav")
            st.success(f"Predicted Emotion: *{emotion}*")

    elif option == "Record audio":
        st.write("Click the record button to record your voice.")
        audio_bytes = st.audio(data=None, format='audio/wav')
        if audio_bytes:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_bytes)
            with st.spinner("Analyzing..."):
                emotion = predict_emotion("temp_audio.wav")
            st.success(f"Predicted Emotion: *{emotion}*")

if __name__ == "__main__":
    main()