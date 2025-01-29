import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import numpy as np
import threading
import soundfile as sf
from datetime import datetime
from Decide import predict_image as decide
import os
from Plot_Spectrograms import plot_user_input as plt_spc


def check_if_allowed():
    return True

class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder")

        self.is_recording = False
        self.recording_thread = None
        self.audio_data = [] 

        self.allowed = check_if_allowed()

        self.label = tk.Label(root, text="Record or Upload" if self.allowed else "Not Allowed", font=("Arial", 14))
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
        with sd.InputStream(callback=callback, channels=1):
            while self.is_recording:
                sd.sleep(100)
        self.save_audio()

    

    def save_audio(self):

        audio_np = np.concatenate(self.audio_data, axis=0)
        print("Recording finished, total frames:", len(audio_np))  # Placeholder action

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        file_path = os.path.join('UserAudio', f'recording_{timestamp}.wav')

        sf.write(file_path, audio_np, 44100)  
        print(f"Audio saved to {file_path}")
        self.check(file_path)

    def upload_file(self):
        global model
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            print(f"Selected file: {file_path}")
        self.check(file_path)
            
    def check(self, file_path):
        global model
        plt_spc(file_path,"UserAudio/Spectrograms")
        result = decide("UserAudio/Spectrograms/user_input.png")
        print(result)
        self.update_label(result)
        os.remove("UserAudio/Spectrograms/user_input.png")


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