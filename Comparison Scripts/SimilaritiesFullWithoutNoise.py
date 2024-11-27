import os
import csv
from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch

# Directory containing spectrograms
spectrogram_dir = 'ColoredSpectrograms_MEL_NoiseReduced'

# Prepare to store results
results = []

# Get all subdirectories
subdirs = [os.path.join(spectrogram_dir, d) for d in os.listdir(spectrogram_dir) if os.path.isdir(os.path.join(spectrogram_dir, d))]

# Open CSV file to write results
with open('spectrogram_similarities_colored_withoutnoise.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Folder', 'Spectrogram1', 'Spectrogram2', 'Similarity'])

    # Iterate through each subdirectory
    for subdir in subdirs:
        folder_name = os.path.basename(subdir)
        spectrogram_files = [f for f in os.listdir(subdir) if f.endswith('.png')]
        
        # Read spectrograms
        spectrograms = [Image.open(os.path.join(subdir, file)) for file in spectrogram_files]
        
        # Compute pairwise similarities
        for i in range(len(spectrogram_files)):
            for j in range(i + 1, len(spectrogram_files)):
                img1 = spectrograms[i]
                img2 = spectrograms[j]
                
                # Calculate pixelmatch similarity
                diff_pixels = pixelmatch(img1, img2, includeAA=True)
                total_pixels = img1.size[0] * img1.size[1]
                similarity = 1 - (diff_pixels / total_pixels)
                
                writer.writerow([folder_name, spectrogram_files[i], spectrogram_files[j], similarity])
                print(f"Processed {spectrogram_files[i]} and {spectrogram_files[j]} in folder {folder_name} with similarity {similarity:.4f}")

print("Similarity calculations complete. Results saved to spectrogram_similarities_colored.csv")