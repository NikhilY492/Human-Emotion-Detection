import os
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model

# --------- CONFIG ---------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "speech_emotion_model.keras")

SAMPLE_RATE = 22050
DURATION = 3  # seconds of audio from mic
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
N_MFCC = 40

emotion_labels = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

# --------- FEATURE EXTRACTION (same as training) ---------

def extract_features_from_raw(audio, sr=SAMPLE_RATE, max_len=SAMPLES_PER_FILE):
    # Ensure 1D
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Trim or pad
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        pad_width = max_len - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    # (n_mfcc, time) -> (1, n_mfcc, time, 1)
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    return mfcc

# --------- MIC RECORDING ---------

def record_from_mic(duration=DURATION, sr=SAMPLE_RATE):
    print(f"\nRecording {duration} seconds from mic...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    audio = recording.flatten()
    return audio

# --------- MAIN ---------

if __name__ == "__main__":
    print("Loading speech emotion model...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    print("\nPress Enter to record, or type 'q' then Enter to quit.")

    while True:
        cmd = input("\n[Enter] to record, [q] to quit: ").strip().lower()
        if cmd == "q":
            print("Exiting.")
            break

        # 1) Record audio from mic
        audio = record_from_mic()

        # 2) Extract features
        x = extract_features_from_raw(audio)

        # 3) Predict
        preds = model.predict(x)
        idx = int(np.argmax(preds))
        emotion = emotion_labels[idx]

        print(f"Predicted emotion: {emotion}  (raw probs: {preds[0]})")
