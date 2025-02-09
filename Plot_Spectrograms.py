import argparse
from matplotlib import pyplot as plt
import soundfile as sf
import pathlib as pl
import librosa
import numpy as np
from datetime import datetime
import noisereduce as nr
import pandas as pd
import os
import random
import librosa.display

def plot_user_input(input_path, output_path):
    rec_dur = 3  # Fixed duration of 3 seconds
    target_sr = 44100  # Fixed sampling rate
    n_mels = 128  # Number of Mel bands
    fmax = 8000  # Maximum frequency for the Mel spectrogram
    img_width = 1000  
    img_height = 400  
    
    # Get audio duration
    duration = librosa.get_duration(path=input_path)
    if duration < rec_dur:
        print("Audio too short for analysis.")
        return False

    # Load and preprocess the audio signal
    signal, original_sr = sf.read(input_path)
    signal = librosa.resample(signal, orig_sr=original_sr, target_sr=target_sr)  # Resample to target_sr

    # Trim leading and trailing silence
    signal, _ = librosa.effects.trim(signal)

    # Split the signal into non-silent intervals and concatenate them
    non_silent_intervals = librosa.effects.split(signal, top_db=20)
    processed_segments = []
    for start, end in non_silent_intervals:
        segment = signal[start:end]
        processed_segments.append(segment)
    signal = np.concatenate(processed_segments)  # Combine segments

    # Check if the concatenated signal is less than 3 seconds
    if len(signal) < rec_dur * target_sr:
        print("Concatenated audio too short for analysis.")
        return False

    # Extract 3-second segment if longer
    if len(signal) > rec_dur * target_sr:
        max_start = len(signal) - rec_dur * target_sr
        start = random.randint(0, max_start)
        signal = signal[start:start + rec_dur * target_sr]
        
    # Reduce noise
    reduced_noise = nr.reduce_noise(y=signal, sr=target_sr)

    # Create Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=reduced_noise, sr=target_sr, n_mels=n_mels, fmax=fmax)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Save the spectrogram as a fixed-size image
    final_output_path = os.path.join(output_path, "user_input.png")
    plt.figure(figsize=(img_width / 100, img_height / 100))  # Convert pixel dimensions to inches (dpi=100)
    plt.axis('off')  # Remove axes
    plt.margins(0, 0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Remove padding
    librosa.display.specshow(mel_spectrogram_db, sr=target_sr, x_axis='time', y_axis='mel', fmax=fmax, cmap='gray_r')
    plt.savefig(final_output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    print(f"Saved spectrogram: {final_output_path}")
    return True
def plot_spectrogram(signal, output_path, interval_index, f, file_name):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=44100, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Save the Mel spectrogram as an image without title and borders
    plt.figure(figsize=(10, 4))
    plt.axis('off')  # Remove axes
    plt.margins(0, 0)  # Remove margins
    plt.gca().set_axis_off()  # Remove axis lines
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Remove padding
    plt.gcf().set_size_inches(10, 4)  # Set figure size
    librosa.display.specshow(mel_spectrogram_db, sr=44100, x_axis='time', y_axis='mel', fmax=8000, cmap='gray_r')
    plt.savefig(f"{output_path / f'{file_name}_i{interval_index}.png'}", bbox_inches='tight', pad_inches=0, format='png')
    plt.close()

    # Write the file path to the output file
    f.write(f"{output_path / f'{file_name}_i{interval_index}.png'},0\n")  # Change it to 1 when it's creating allowed class

    return mel_spectrogram_db


def main(file_path, file_name):

    joined_path = str(os.path.join(file_path, file_name))
    print(f"Original audio duration: {librosa.get_duration(path=joined_path)} seconds")
    signal, signal_rate = sf.read(joined_path)
    f = open("MEL_Spectograms/labels.csv", "a")
    # Trim silence
    signal, _ = librosa.effects.trim(signal, top_db=20)
    print(f"Audio duration after trimming silence: {len(signal) / signal_rate:.2f} seconds")
    non_silent_intervals = librosa.effects.split(signal, top_db=20)
    processed_segments = []
    for start, end in non_silent_intervals:
        segment = signal[start:end]
        processed_segments.append(segment)
    signal = np.concatenate(processed_segments)  # Combine segments

    reduced_noise = nr.reduce_noise(signal, sr=signal_rate)
    signal = reduced_noise
    print(f"Audio duration after reducing noise: {len(signal) / signal_rate:.2f} seconds")

    #Split into 3-second intervals
    interval_duration = 3 * signal_rate
    num_intervals = int(np.floor(len(signal) / interval_duration))

    for i in range(num_intervals):
        start = i * interval_duration
        end = min((i + 1) * interval_duration, len(signal))
        interval_signal = signal[start:end]

        print(
            f"Processing interval {i + 1}/{num_intervals}, duration: {len(interval_signal) / signal_rate:.2f} seconds")
        spectrogram_db = plot_spectrogram(
            interval_signal, pl.Path('MEL_Spectograms/images'), i + 1, f, file_name)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Spectrogram')
    parser.add_argument('audiofile_path', type=str, help='Path to the audio file')
    parser.add_argument('audiofile_name', type=str, help='Name of the audio file')
    args = parser.parse_args()
    main(args.audiofile, args.audiofile_name)
