import sounddevice as sd
import numpy as np
import os
import time
import pickle
import torch
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import scipy.io.wavfile as wav

# Load the Speaker Recognition Model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                            run_opts={"device":"cpu"}, 
                                            savedir="tmp_model")

# Directory for storing speaker embeddings
SPEAKER_DB = "speakers_db.pkl"  # File where we store known speakers
AUDIO_DIR = "recordings"
SAMPLE_RATE = 16000
THRESHOLD = 0.01  # Adjust based on environment
GRACE_PERIOD = 2  # Allow pause time before stopping recording

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True) 

# Load or initialize the speaker database
if os.path.exists(SPEAKER_DB):
    with open(SPEAKER_DB, "rb") as f:
        speaker_profiles = pickle.load(f)
else:
    speaker_profiles = {}

# Global variables
recording = []
is_recording = False
last_sound_time = None


def extract_speaker_embedding(audio_path):
    """Extracts speaker embedding from an audio file."""
    signal = sb.dataio.dataio.read_audio(audio_path).unsqueeze(0)
    embedding = classifier.encode_batch(signal).squeeze(0).detach().numpy()
    return embedding


def match_speaker(embedding, threshold=0.6):
    """Matches the given embedding with known speakers."""
    best_match = None
    best_score = float("inf")

    for name, stored_embedding in speaker_profiles.items():
        distance = np.linalg.norm(embedding - stored_embedding)  # Euclidean distance
        if distance < best_score:
            best_score = distance
            best_match = name

    if best_score < threshold:
        return best_match
    return None


def save_new_speaker(audio_path):
    """Asks the user to label a new speaker and saves their embedding."""
    embedding = extract_speaker_embedding(audio_path)

    print("\nNew speaker detected! Please enter a name:")
    speaker_name = input("Speaker Name: ").strip()

    if speaker_name:
        speaker_profiles[speaker_name] = embedding
        with open(SPEAKER_DB, "wb") as f:
            pickle.dump(speaker_profiles, f)
        print(f"Speaker '{speaker_name}' saved successfully!")


def save_recording(audio_data):
    """Saves the recorded audio and processes the speaker's identity."""
    if not audio_data:
        return

    audio_array = np.concatenate(audio_data, axis=0).flatten()
    if len(audio_array) == 0:
        return

    timestamp = int(time.time())
    filename = os.path.join(AUDIO_DIR, f"recording_{timestamp}.wav")
    wav.write(filename, SAMPLE_RATE, (audio_array * 32767).astype(np.int16))
    print(f"Recording saved: {filename}")

    # Ensure the file is fully written
    time.sleep(5)  # Short delay to ensure file is ready

    # Extract speaker embedding
    print('Extracting embeddings')
    embedding = extract_speaker_embedding(filename)

    # Try to match the speaker
    print('Matching speaker')
    matched_speaker = match_speaker(embedding)

    if matched_speaker:
        print(f"Speaker recognized: {matched_speaker}")
    else:
        print("Unknown speaker detected.")
        save_new_speaker(filename)  # Ask the user to label this speaker


def audio_callback(indata, frames, time_info, status):
    """Processes real-time audio and detects speech."""
    global recording, is_recording, last_sound_time

    volume_norm = np.linalg.norm(indata) / frames

    if volume_norm > THRESHOLD:
        if not is_recording:
            print("Sound detected! Starting recording...")
            is_recording = True
        recording.append(indata.copy())
        last_sound_time = time.time()
    elif is_recording:
        if time.time() - last_sound_time > GRACE_PERIOD:
            print("Silence detected, stopping recording...")
            save_recording(recording)
            recording = []
            is_recording = False


def main_loop():
    """Main loop to handle real-time audio detection."""
    global last_sound_time

    print("Listening for speakers...")
    with sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=1):
        last_sound_time = time.time()
        while True:
            time.sleep(0.1)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nStopping the application.")
