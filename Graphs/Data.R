data <- read.csv('Graphs/TablePixelComparisonFullLengthSameUserNoiseReduce.csv')
labels <- read.csv('MEL_Spectograms/labels.csv', header = FALSE)

data_filtered <- data[data$Similarity > 0.95,]
nrow(data_filtered)

str(data_filtered)

data_filtered$Pair <- paste(pmin(data_filtered$Spectrogram1, data_filtered$Spectrogram2),
                            pmax(data_filtered$Spectrogram1, data_filtered$Spectrogram2),
                            sep = "_")

duplicated_pairs <- data_filtered[duplicated(data_filtered$Pair) | duplicated(data_filtered$Pair, fromLast = TRUE), ]

str(labels)

data_filtered$Spectrogram2 <- gsub("\\.png$", "", data_filtered$Spectrogram2)

data_filtered$Spectrogram2

# Extract the base names from Spectrogram2
base_names <- unique(data_filtered$Spectrogram2)

# Create a new variable labels_filtered that excludes any rows where the file path includes any value from Spectrogram2
labels_filtered <- labels[!sapply(labels[, 1], function(x) any(sapply(base_names, function(y) grepl(y, x)))), ]

library(dplyr)

labels <- labels %>%
  mutate(wav_file = sub(".*/(.*?\\.wav)_.*", "\\1", V1))

# Filter out rows in labels where the WAV file exists in csv2
filtered_labels <- labels %>%
  filter(!wav_file %in% data_filtered$Spectrogram2) %>%
  select(-wav_file) # Remove the temporary column

# Save the filtered data
write.csv(filtered_labels, "filtered_labels.csv", row.names = FALSE)


print("Filtered labels:")
print(labels_filtered)

nrow(labels_filtered)

training_labels <- labels %>%
  filter(grepl("script[1-4]", V1))

training_labels

testing_labels <- labels %>%
  filter(grepl("script5", V1))

testing_labels

write.csv(training_labels, "training_labels.csv", row.names = FALSE)
write.csv(testing_labels, "testing_labels.csv", row.names = FALSE)

