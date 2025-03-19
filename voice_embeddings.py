from resemblyzer import VoiceEncoder, preprocess_wav
import numpy

def resemblyzer_voice_embeddings(audio_path: str) -> numpy.ndarray:
    """
    Extracts voice embeddings from an audio file using the Resemblyzer model.

    Parameters:
        audio_path (str): Path to the input audio file.

    Returns:
        numpy.ndarray: A NumPy array containing the voice embedding.
    """
    
    # Preprocess the audio file (convert to the required format)
    wav = preprocess_wav(audio_path)
    
    # Initialize the voice encoder
    encoder = VoiceEncoder()
    
    # Extract and return the voice embedding
    return encoder.embed_utterance(wav)