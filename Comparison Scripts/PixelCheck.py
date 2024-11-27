import os
import csv
import concurrent.futures
from skimage import io
from pixelmatch.contrib.PIL import pixelmatch
from PIL import Image

# Directory containing spectrograms
spectrogram_dir = 'MEL_Spectograms/images'
output_file = 'spectrogram_similarities_pixel_check.csv'

# Prepare to store results and load existing entries
existing_results = set()
if os.path.exists(output_file):
    with open(output_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            existing_results.add((row[0], row[1], row[2]))

# Group spectrograms by recording
spectrogram_files = [f for f in os.listdir(spectrogram_dir) if f.endswith('.png')]
recording_groups = {}
for filename in spectrogram_files:
    recording_id = "_".join(filename.split('_')[:2])  # e.g., "f1_script1"
    recording_groups.setdefault(recording_id, []).append(filename)

# Function to calculate similarity between two spectrograms
def calculate_similarity(recording_id, file1, file2):
    img1 = Image.open(os.path.join(spectrogram_dir, file1))
    img2 = Image.open(os.path.join(spectrogram_dir, file2))
    diff_pixels = pixelmatch(img1, img2, includeAA=True)
    total_pixels = img1.size[0] * img1.size[1]
    similarity = 1 - (diff_pixels / total_pixels)
    return (recording_id, file1, file2, similarity)

# Open CSV file to write results
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not existing_results:
        writer.writerow(['Recording', 'Spectrogram1', 'Spectrogram2', 'Similarity'])  # Add header if file is new

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        # Schedule similarity calculations for each pair
        for recording_id, files in recording_groups.items():
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    pair_key = (recording_id, files[i], files[j])
                    if pair_key not in existing_results:  # Skip if already processed
                        futures.append(
                            executor.submit(calculate_similarity, recording_id, files[i], files[j])
                        )

        # Write results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            writer.writerow(result)
            print(f"Processed {result[1]} and {result[2]} with similarity {result[3]:.4f}")

print("Similarity calculations complete. Results saved to spectrogram_similarities_pixel_check.csv")
