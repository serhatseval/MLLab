import argparse
from matplotlib import pyplot as plt
import soundfile as sf
import pathlib as pl
import librosa
import numpy as np
from datetime import datetime


def plot_spectrogram(signal, signal_rate, output_path, interval_index, f):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    plt.figure(figsize=(16, 10), dpi=100, constrained_layout=True)
    plt.axis('off')
    img = librosa.display.specshow(spectrogram_db, sr=signal_rate, x_axis=None, y_axis='log', cmap='Grays')
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    plt.savefig(output_path / f'spectrogram_{timestamp}_interval_{interval_index}.png')
    f.write(f"{output_path / f'spectrogram_{timestamp}_interval_{interval_index}.png'},1\n")
    plt.close()
    return spectrogram_db

def main(file_path):
    print(f"Original audio duration: {librosa.get_duration(path=file_path)} seconds")
    signal, signal_rate = sf.read(file_path)
    f = open("testData/labels.csv", "a")
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
        spectrogram_db = plot_spectrogram(interval_signal, signal_rate, pl.Path('OutputFiles/'), i + 1,f)
        print(f"Spectrogram Shape: {spectrogram_db.shape}")
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Spectrogram')
    parser.add_argument('audiofile', type=str, help='Path to the audio file')
    args = parser.parse_args()
    main(args.audiofile)
