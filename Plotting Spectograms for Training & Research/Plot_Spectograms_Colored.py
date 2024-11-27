import argparse
from matplotlib import pyplot as plt
import soundfile as sf
import pathlib as pl
import librosa
import numpy as np
import noisereduce as nr
import os

def plot_mel_spectrogram(signal, sr, output_path, file_name):
    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Ensure the size of the spectrogram is consistent
    plt.figure(figsize=(10, 4))
    plt.axis('off')  # Remove axes
    plt.margins(0, 0)  # Remove margins
    plt.gca().set_axis_off()  # Remove axis lines
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Remove padding
    plt.gcf().set_size_inches(10, 4)  # Set figure size
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis')
    plt.savefig(output_path / f"{file_name}.png", bbox_inches='tight', pad_inches=0, format='png')
    plt.close()
    return mel_spectrogram_db

def main(file_path, file_name):
    joined_path = str(os.path.join(file_path, file_name))
    print(f"Original audio duration: {librosa.get_duration(path=joined_path)} seconds")
    signal, sigrat = sf.read(joined_path)
    
    # Determine the output directory based on the filename's starting identifiers
    identifier = file_name[:3] if file_name[:2].isdigit() else file_name[:2]
    output_dir = pl.Path(f'ColoredSpectrograms_MEL_NoiseReduced/{identifier}')
    output_dir.mkdir(parents=True, exist_ok=True)
    reduced_noise = nr.reduce_noise(signal, sr=sigrat)
    signal = reduced_noise

    # Plot and save the Mel spectrogram
    plot_mel_spectrogram(signal, sigrat, output_dir, file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Mel Spectrogram')
    parser.add_argument('audiofile_path', type=str, help='Path to the audio file')
    parser.add_argument('audiofile_name', type=str, help='Name of the audio file')
    args = parser.parse_args()
    main(args.audiofile_path, args.audiofile_name)