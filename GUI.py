import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import numpy as np
import threading
import soundfile as sf
from datetime import datetime
import os
from Decide import load_model, decide 
from plot_spectogram import plot_user_input

global model

def check_if_allowed():
    return True

class RecorderApp:
    def __init__(self, root):
        model = load_model("OutputFiles/model10.pth")
        self.root = root
        self.root.title("Audio Recorder")

        self.is_recording = False
        self.recording_thread = None
        self.audio_data = [] 

        self.allowed = check_if_allowed()

        self.label = tk.Label(root, text="Allowed" if self.allowed else "Not Allowed", font=("Arial", 14))
        self.label.pack(pady=20)

        self.record_button = tk.Button(root, text="Start Recording", command=self.toggle_recording, width=20)
        self.record_button.pack(pady=20)

        self.upload_button = tk.Button(root, text="Upload WAV File", command=self.upload_file, width=20)
        self.upload_button.pack(pady=20)

        if not self.allowed:
            self.record_button.config(state="disabled")
            self.label.config(text="Not Allowed")

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
        else:
            self.is_recording = False
            self.record_button.config(text="Start Recording")

    def record_audio(self):
        # Function to capture audio
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_data.append(indata.copy())

        # Start recording with the sounddevice library
        with sd.InputStream(callback=callback):
            while self.is_recording:
                sd.sleep(100)
        self.save_audio()

    def save_audio(self):
        audio_np = np.concatenate(self.audio_data, axis=0)
        print("Recording finished, total frames:", len(audio_np))  # Placeholder action

        if not os.path.exists('recordedaudio'):
            os.makedirs('recordedaudio')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        file_path = os.path.join('recordedaudio', 'spectograms',f'recording_{timestamp}.png')
        sf.write(file_path, audio_np, 44100)  # Assuming a sample rate of 44100 Hz
        print(f"Audio saved to {file_path}")
        spectogram_path = os.path.join('recordedaudio','spectograms', f'spectogram_{timestamp}.png')
        plot_user_input(file_path, spectogram_path)
        result = decide(file_path, spectogram_path)
        os.remove(file_path)
        os.remove(spectogram_path)
        self.update_label(result)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            print(f"Selected file: {file_path}")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            spectogram_path = os.path.join('uploadedaudio', f'spectogram_{timestamp}.png')
            plot_user_input(file_path, spectogram_path)
            result = decide(file_path, spectogram_path)
            self.update_label(result)

    def update_label(self, result):
        if result == 0:
            self.label.config(text="Not Allowed Category:0")
        elif result == 1:
            self.label.config(text="Allowed Category:1")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("250x250")
    root.eval('tk::PlaceWindow . center')
    app = RecorderApp(root)
    root.mainloop()