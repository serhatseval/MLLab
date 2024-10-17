import argparse
from matplotlib import pyplot as plt
import soundfile as sf
import pathlib as pl
import librosa
import numpy as np
from datetime import datetime

def plot_spectogram(signal, signal_rate, outputpath):
    stft = librosa.stft(signal)
    spectogram = np.abs(stft)
    spectogram_db = librosa.amplitude_to_db(spectogram)
    plt.figure(figsize=(14, 5))
    img = librosa.display.specshow(spectogram_db, sr=signal_rate, x_axis= 'time', y_axis='log', cmap='Grays')
    outputpath.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(outputpath / f'spectogram_{timestamp}.png')
    plt.show()
    return spectogram_db

def main(audiofile): 
    signal, signal_rate = sf.read(audiofile)
    print(f"Signal Rate: {signal_rate}")
    spectrogram_db = plot_spectogram(signal, signal_rate, pl.Path('OutputFiles/'))
    print(f"Audio Duration: {len(signal)/signal_rate:.2f} seconds")
    print(f"Spectrogram Shape: {spectrogram_db.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Spectogram')
    parser.add_argument('audiofile', type=str, help='Path to the audio file')
    args = parser.parse_args()
    main(args.audiofile)
