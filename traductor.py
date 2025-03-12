import whisper

def transcribe_audio(audio_path: str, model_size: str = "large") -> str:
    """
    Transcribes and translates an audio file using OpenAI's Whisper model.

    Parameters:
        audio_path (str): Path to the input audio file.
        model_size (str): Size of the Whisper model to use (default: "large").

    Returns:
        str: The translated text.
    """
    # Load the Whisper model
    model = whisper.load_model(model_size)

    # Transcribe and translate the audio
    result = model.transcribe(audio_path)

    return result["text"]

# Example usage:
# translated_text = transcribe_audio("recordings/recording_1741561011.wav")
# print(translated_text)