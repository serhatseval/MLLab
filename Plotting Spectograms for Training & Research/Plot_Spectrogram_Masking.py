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


def time_masking(mel_spectrogram_db, max_mask_percentage=0.2):
    time_steps = mel_spectrogram_db.shape[1]
    mask_size = int(time_steps * max_mask_percentage)
    start = random.randint(0, time_steps - mask_size)
    mel_spectrogram_db[:, start:start+mask_size] = 0
    return mel_spectrogram_db


def plot_spectrogram(signal, output_path, interval_index, f, file_name, classification, timemask):
    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=44100, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if timemask:
        mel_spectrogram_db = time_masking(mel_spectrogram_db)

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
    f.write(f"{output_path / f'{file_name}_i{interval_index}.png'},{classification}\n")

    return mel_spectrogram_db



def main(file_path, file_name, classification):

    joined_path = str(os.path.join(file_path, file_name))
    print(f"Original audio duration: {librosa.get_duration(path=joined_path)} seconds")
    signal, signal_rate = sf.read(joined_path)
    f = open("MEL_Spectograms_TimeMasking_20/labels.csv", "a")
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
        
        apply_mask = random.random() < 0.2  # 50% chance of time masking
        if apply_mask:
            plot_spectrogram(interval_signal, pl.Path('MEL_Spectograms_TimeMasking_20/images'), i + 1, f, file_name, classification,1)
        else:
            plot_spectrogram(interval_signal, pl.Path('MEL_Spectograms_TimeMasking_20/images'), i + 1, f, file_name, classification,0)

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Spectrogram')
    parser.add_argument('audiofile_path', type=str, help='Path to the audio file')
    parser.add_argument('audiofile_name', type=str, help='Name of the audio file')
    args = parser.parse_args()
    main(args.audiofile, args.audiofile_name)
