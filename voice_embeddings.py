from resemblyzer import VoiceEncoder, preprocess_wav
import numpy

def resemblyzer_voice_embeddings(audio_path: str, encoder) -> numpy.ndarray:
    """
    Extracts voice embeddings from an audio file in audio_path using the Resemblyzer model.

    Parameters:
        audio_path (str): Path to the input audio file.
        encoder: resemblyzer voice encoder instance

    Returns:
        numpy.ndarray: A NumPy array containing the voice embedding.
    """
    
    # Preprocess the audio file (convert to the required format)
    wav = preprocess_wav(audio_path)
    
    # Extract and return the voice embedding
    return encoder.embed_utterance(wav)

def resemblyzer_speaker_embeddings(audio_paths: list, encoder) -> numpy.ndarray:
    """
    Extracts speaker embeddings from an audio file in audio_path using the Resemblyzer model.

    Parameters:
        audio_path (str): Path to the input audio file.
        encoder: resemblyzer voice encoder instance

    Returns:
        numpy.ndarray: A NumPy array containing the voice embedding.
    """
    
    # Preprocess the audio file (convert to the required format)
    wav = [preprocess_wav(i) for i in audio_paths]
    
    # Extract and return the speaker embedding
    return encoder.embed_speaker(wav)