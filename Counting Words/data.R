script1<-read.csv("csv_files/script1_count.csv")
script2<-read.csv("csv_files/script2_count.csv")
script3<-read.csv("csv_files/script3_count.csv")
script4<-read.csv("csv_files/script4_count.csv")
script5<-read.csv("csv_files/script5_count.csv")

script1<- script1[order(-script1$Count),]
script2<- script2[order(-script2$Count),]
script3<- script3[order(-script3$Count),]
script4<- script4[order(-script4$Count),]
script5<- script5[order(-script5$Count),]

num_rows <- c(nrow(script1), nrow(script2), nrow(script3), nrow(script4), nrow(script5))
script_names <- c("Script 1", "Script 2", "Script 3", "Script 4", "Script 5")

# Create a data frame for plotting
data <- data.frame(Script = script_names, Rows = num_rows)

data

# Save the plot as a PNG file
png("number_of_rows_plot.png")
barplot(data$Rows, names.arg = data$Script, col = "skyblue", xlab = "Script", ylab = "Number of Rows", main = "Number of Rows in Each Script")
dev.off()
