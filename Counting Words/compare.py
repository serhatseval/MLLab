import os
import pandas as pd
from scipy.spatial.distance import cosine
import csv

def read_csv_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {}
    for file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, file), index_col=0)
        dataframes[file] = df
    return dataframes

def calculate_similarity(df1, df2):
    words = set(df1.index).union(set(df2.index))
    vec1 = [df1.loc[word].values[0] if word in df1.index else 0 for word in words]
    vec2 = [df2.loc[word].values[0] if word in df2.index else 0 for word in words]
    return 1 - cosine(vec1, vec2)

def calculate_all_similarities(dataframes):
    similarities = []
    files = list(dataframes.keys())
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            similarity = calculate_similarity(dataframes[files[i]], dataframes[files[j]])
            similarities.append((files[i], files[j], similarity))
    return similarities

def remove_suffix(filename, suffix="_count.csv"):
    if filename.endswith(suffix):
        return filename[:-len(suffix)]
    return filename

folder_path = '/Users/serhatseval/Downloads/ml_count_words/csv_files'
dataframes = read_csv_files(folder_path)
similarities = calculate_all_similarities(dataframes)


results_folder = '/Users/serhatseval/Downloads/ml_count_words/results'
os.makedirs(results_folder, exist_ok=True)

with open(os.path.join(results_folder, 'similarities.txt'), 'w') as f:
    for file1, file2, similarity in similarities:
        file1_clean = remove_suffix(file1)
        file2_clean = remove_suffix(file2)
        f.write(f"{file1_clean} --- {file2_clean}: {similarity:.4f}\n")

with open(os.path.join(results_folder, 'similarities.csv'), 'w', newline='') as csvfile:
    fieldnames = ['file1', 'file2', 'similarity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for file1, file2, similarity in similarities:
        file1_clean = remove_suffix(file1)
        file2_clean = remove_suffix(file2)
        writer.writerow({'file1': file1_clean, 'file2': file2_clean, 'similarity': similarity})
