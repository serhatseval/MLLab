import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import threading


# Function to check if recording is allowed (this could be modified as per requirements)
def check_if_allowed():
    # For demonstration, this could be a more complex check in real scenarios
    return True


class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder")

        self.is_recording = False
        self.recording_thread = None
        self.audio_data = []  # Store the recorded audio

        # Label to indicate recording status
        self.label = tk.Label(root, text="Allowed" if check_if_allowed() else "Not Allowed", font=("Arial", 14))
        self.label.pack(pady=20)

        # Start/Stop recording button
        self.record_button = tk.Button(root, text="Start Recording", command=self.toggle_recording, width=20)
        self.record_button.pack(pady=20)

        # Flag to check if recording is allowed
        self.allowed = check_if_allowed()

        if not self.allowed:
            self.record_button.config(state="disabled")
            self.label.config(text="Not Allowed")

    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
        else:
            # Stop recording
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
        # Process to save or handle the recorded audio, this can be modified as needed
        audio_np = np.concatenate(self.audio_data, axis=0)
        print("Recording finished, total frames:", len(audio_np))  # Placeholder action


if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()
