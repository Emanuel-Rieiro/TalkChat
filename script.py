import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time
import threading

# Parameters
SAMPLE_RATE = 16000  # Sample rate in Hz
THRESHOLD = 0.01  # Initial sound energy threshold
CHUNK_DURATION = 0.5  # Duration of each audio chunk in seconds
OUTPUT_DIR = "recordings"  # Directory to save recordings
GRACE_PERIOD = 2  # Time (in seconds) to keep recording after sound stops
SMOOTHING_FACTOR = 0.3  # Weight for smoothing volume detection

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global Variables
recording = []  # Buffer for audio data
is_recording = False  # Recording status
last_sound_time = None  # Time of the last detected sound
average_volume = 0  # Smoothed average volume
lock = threading.Lock()  # For thread-safe operations


def audio_callback(indata, frames, time_info, status):
    """
    This callback processes audio chunks captured by the microphone.
    """
    global recording, is_recording, last_sound_time, average_volume

    # Compute the volume of the current chunk (RMS value)
    volume_norm = np.sqrt(np.mean(indata**2))

    # Update the smoothed average volume
    average_volume = SMOOTHING_FACTOR * volume_norm + (1 - SMOOTHING_FACTOR) * average_volume

    # Debugging: Log the volume level (optional)
    print(f"Current volume: {volume_norm:.4f}, Smoothed volume: {average_volume:.4f}")

    if average_volume > THRESHOLD:
        # Sound detected
        with lock:
            if not is_recording:
                print("Sound detected! Starting recording...")
                is_recording = True
            recording.append(indata.copy())  # Add chunk to recording buffer
            last_sound_time = time.time()  # Update the time of the last sound
    elif is_recording:
        # Silence detected
        current_time = time.time()
        if current_time - last_sound_time > GRACE_PERIOD:
            # Grace period exceeded: stop recording
            print("Silence detected for grace period. Saving recording...")
            save_recording(recording)
            recording = []  # Clear buffer
            is_recording = False


def save_recording(audio_data):
    """
    Save the recorded audio to a WAV file if it contains valid data.
    """
    if not audio_data:
        return  # No audio data to save

    audio_array = np.concatenate(audio_data, axis=0).flatten()
    if len(audio_array) == 0:
        return  # Avoid saving empty recordings

    timestamp = int(time.time())
    filename = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.wav")
    wav.write(filename, SAMPLE_RATE, (audio_array * 32767).astype(np.int16))  # Convert to int16 format
    print(f"Recording saved: {filename}")


def main_loop():
    """
    Main loop to handle recording logic and silence detection.
    """
    #global recording, is_recording, last_sound_time

    print("Listening for sounds...")
    with sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=1):
        last_sound_time = time.time()  # Initialize the silence timer

        while True:
            time.sleep(0.1)  # Check every 100ms


if __name__ == "__main__":
    try:
        # Run the main loop in the main thread
        main_loop()
    except KeyboardInterrupt:
        print("\nStopping the app.")
