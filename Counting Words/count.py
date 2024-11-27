import os
import csv
import string
from collections import Counter

def count_words_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        text = text.lower()  # Convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        words = text.split()
        word_count = Counter(words)
        return word_count

def save_word_counts_to_csv(word_counts, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Word', 'Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for word, count in word_counts.items():
            writer.writerow({'Word': word, 'Count': count})

def main():
    input_folder = './txt_files'  
    output_folder = './csv_files'  

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            word_counts = count_words_in_file(file_path)
            output_csv = os.path.join(output_folder, os.path.splitext(filename)[0] + '_count.csv')
            save_word_counts_to_csv(word_counts, output_csv)

if __name__ == "__main__":
    main()