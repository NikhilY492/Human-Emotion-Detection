import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

# ------------------- CONFIG -------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "ravdess", "Audio_Speech_Actors_01-24")

# sample rate & duration
SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION

# number of MFCC features
N_MFCC = 40

# emotions map (only using 8 emotions from RAVDESS)
emotion_map = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7,  # surprised
}

NUM_EMOTIONS = len(emotion_map)

# ------------------- FEATURE EXTRACTION -------------------

def extract_features(file_path, max_len=SAMPLES_PER_FILE):
    # load audio
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # trim or pad to fixed length
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        pad_width = max_len - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')

    # compute MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    # normalize (per feature)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    # shape: (n_mfcc, time) -> we'll add channel dimension later
    return mfcc

def load_ravdess(data_dir=DATA_DIR):
    X = []
    y = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            file_path = os.path.join(root, file)

            # filename example: 03-01-05-01-02-01-12.wav
            parts = file.split("-")
            if len(parts) < 3:
                continue

            emotion_code = parts[2]
            if emotion_code not in emotion_map:
                continue

            label = emotion_map[emotion_code]

            mfcc = extract_features(file_path)
            X.append(mfcc)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # add channel dimension for Conv2D: (samples, n_mfcc, time, 1)
    X = np.expand_dims(X, -1)

    y = to_categorical(y, num_classes=NUM_EMOTIONS)

    return X, y

# ------------------- BUILD MODEL -------------------

def build_model(input_shape, num_classes=NUM_EMOTIONS):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ------------------- MAIN TRAINING -------------------

if __name__ == "__main__":
    print("Loading RAVDESS data...")
    X, y = load_ravdess()
    print("Data shape:", X.shape, "Labels shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    input_shape = X_train.shape[1:]  # (n_mfcc, time, 1)
    model = build_model(input_shape)

    print(model.summary())

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=32
    )

    # save weights & model
    model.save_weights(os.path.join(BASE_DIR, "speech_emotion.weights.h5"))
    model.save(os.path.join(BASE_DIR, "speech_emotion_model.keras"))

    print("Training complete. Models saved.")
