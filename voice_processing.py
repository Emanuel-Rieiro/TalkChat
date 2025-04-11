import torchaudio
import whisper
from resemblyzer import VoiceEncoder, preprocess_wav
from voice_database import VoiceRegistry

class VoiceProcessor:
    def __init__(self, whisper_version = "turbo", registry_file = "test_registry_prespectiva.pkl"):
        # Load speaker embedding model
        self.embedding_model = VoiceEncoder()

        # Load Pyannote diarization model
        self.diarization_model = whisper.load_model(whisper_version)

        # Load voice registry
        self.speaker_registry = VoiceRegistry(filepath = registry_file)

    def process_audio(self, audio_path):
        """
        Processes an audio file to extract speaker embeddings for each unique speaker.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            list: List of dictionaries with the segments of the audio and the recognized speakers
        """

        # Step 1: Diarize to identify speaker segments
        diarization = self.diarization_model.transcribe(audio_path)

        # Step 2: Divide the results from the whisper transcription
        audio_text = diarization['text']
        audio_segments = diarization['segments']

        # Step 3: Load the full audio for slicing
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform[0].numpy()  # Convert to mono

        # Step 4: For each segment, extract embedding
        for i, s, e, t in ((s["id"], s["start"], s["end"], s["text"]) for s in audio_segments):
            print(s, e, t)
            start = int(s * sample_rate)
            end = int(e * sample_rate)
            segment = waveform[start:end]
            audio_segments[i]['speaker'] = ""

            if len(segment) < sample_rate:  # Skip too short segments
                print('Skipped segment', s, e)
                continue
            
            # Get embeddings for the speaker
            segment_embedding = self.embedding_model.embed_speaker([preprocess_wav(segment)])

            # Update segments with speaker
            audio_segments[i]['speaker'] = self.speaker_registry.process_voice(segment_embedding, update = True)

        return audio_segments