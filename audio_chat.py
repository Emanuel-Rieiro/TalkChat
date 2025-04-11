import tkinter as tk
from tkinter import filedialog, scrolledtext
from voice_processing import VoiceProcessor  # assuming this contains your class
import os

class AudioChatUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Chat Viewer")
        self.processor = VoiceProcessor(whisper_version = 'large')

        # UI layout
        self.chat_frame = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=60, height=25, state='disabled', font=("Helvetica", 12))
        self.chat_frame.pack(padx=10, pady=10)

        self.load_button = tk.Button(master, text="Load Audio File", command=self.load_audio)
        self.load_button.pack(pady=(0, 10))

    def load_audio(self):
        audio_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac")])
        if not audio_path:
            return
        
        # Display audio path
        self.append_chat("System", f"Loaded: {os.path.basename(audio_path)}")

        # Process the audio
        speaker_embeddings = self.processor.process_audio(audio_path)

        # Show detected speakers
        for segment in speaker_embeddings:
            self.append_chat(segment['speaker'], segment['text'])

    def append_chat(self, sender, message):
        self.chat_frame.config(state='normal')
        self.chat_frame.insert(tk.END, f"{sender}: {message}\n")
        self.chat_frame.config(state='disabled')
        self.chat_frame.see(tk.END)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioChatUI(root)
    root.mainloop()