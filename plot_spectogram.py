import argparse
from matplotlib import pyplot as plt
import soundfile as sf
import pathlib as pl
import librosa
import numpy as np
from datetime import datetime
import os


def plot_user_input(input_path, output_path):
    duration = librosa.get_duration(path=input_path)
    if duration < 10 or duration > 20:
        return 0
    signal, signal_rate = sf.read(input_path, frames=0)
    start = (duration - 10)*signal_rate//2
    signal, _ = sf.read(input_path, start=start, frames=signal_rate*10)
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    plt.imsave(fname=output_path, arr=spectrogram_db, cmap='gray_r', format='png')


def plot_spectrogram(signal, output_path, interval_index, f, file_name):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    plt.imsave(fname=
               f"{output_path / f'{file_name}_i{interval_index}.png'}", arr=spectrogram_db, cmap='gray_r', format='png')
    f.write(f"{output_path / f'{file_name}_i{interval_index}.png'},0\n")
    plt.close()
    return spectrogram_db


def main(file_path, file_name):
    joined_path = str(os.path.join(file_path, file_name))
    print(f"Original audio duration: {librosa.get_duration(path=joined_path)} seconds")
    signal, signal_rate = sf.read(joined_path)
    f = open("OutputFiles/labels.csv", "a")
    # Trim silence
    signal, _ = librosa.effects.trim(signal)
    print(f"Audio duration after trimming silence: {len(signal) / signal_rate:.2f} seconds")

    # Split into 10-second intervals
    interval_duration = 10 * signal_rate
    num_intervals = int(np.floor(len(signal) / interval_duration))

    for i in range(num_intervals):
        start = i * interval_duration
        end = min((i + 1) * interval_duration, len(signal))
        interval_signal = signal[start:end]

        print(
            f"Processing interval {i + 1}/{num_intervals}, duration: {len(interval_signal) / signal_rate:.2f} seconds")
        spectrogram_db = plot_spectrogram(
            interval_signal, pl.Path('OutputFiles/images'), i + 1, f, file_name)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Spectrogram')
    parser.add_argument('audiofile_path', type=str, help='Path to the audio file')
    parser.add_argument('audiofile_name', type=str, help='Name of the audio file')
    args = parser.parse_args()
    main(args.audiofile, args.audiofile_name)
    # main("C:\\Users\\Marcel\\Desktop\\IML\\daps\\daps\\ClipsForCNN\\ipad_balcony1", "f1_script5_ipad_balcony1.wav")
