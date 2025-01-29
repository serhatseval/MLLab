data <- read.csv('/Users/serhatseval/Developer/MLLab/Comparison Scripts/spectrogram_similarities_colored.csv')

data <- data[order(data$Similarity, decreasing = TRUE), ]

head(data, 10)
