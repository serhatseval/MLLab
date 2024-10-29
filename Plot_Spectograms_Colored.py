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


def plot_spectrogram(signal, output_path, file_name):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    plt.imsave(fname=output_path / f"{file_name}.png", arr=spectrogram_db, cmap='viridis', format='png')
    plt.close()
    return spectrogram_db


def main(file_path, file_name):
    joined_path = str(os.path.join(file_path, file_name))
    print(f"Original audio duration: {librosa.get_duration(path=joined_path)} seconds")
    signal, signal_rate = sf.read(joined_path)
    
    # Determine the output directory based on the filename's starting identifiers
    identifier = file_name[:3] if file_name[:2].isdigit() else file_name[:2]
    output_dir = pl.Path(f'OutputFilesMarta/{identifier}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save the spectrogram
    plot_spectrogram(signal, output_dir, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Spectrogram')
    parser.add_argument('audiofile_path', type=str, help='Path to the audio file')
    parser.add_argument('audiofile_name', type=str, help='Name of the audio file')
    args = parser.parse_args()
    main(args.audiofile_path, args.audiofile_name)