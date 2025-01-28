import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict
import datetime

# Paths
annotations_file = 'MEL_Spectograms/labels_removing_95_percent_similarity.csv'
img_dir = 'MEL_Spectograms/images'

# Constants
img_height = 1000
img_width = 400  # Length for 3 seconds of audio
features = 2
batch_size = 16
epochs = 10

# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Device configuration
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Model creation
def create_model():
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Add dropout layer before the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # 50% dropout rate
        nn.Linear(num_features, features)
    )
    return model.to(device)


# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Validation loop
def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    val_loss /= num_batches
    correct /= size
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    return val_loss, correct, f1, cm

# Save results
def save_results(filename, accuracy, avg_loss, f1_score, cm):
    with open(filename, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}, F1 Score: {f1_score:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")

# Calculate class weights
def calculate_class_weights(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_sample_counts = pd.Series(labels).value_counts().sort_index()
    weights = 1.0 / class_sample_counts
    class_weights = torch.tensor(weights / weights.sum(), dtype=torch.float32).to(device)
    return class_weights

# Group intervals by audio file
def get_audio_file_groups(annotations_file):
    df = pd.read_csv(annotations_file)
    groups = defaultdict(list)
    for idx, row in df.iterrows():
        audio_file = row[0].split('_i')[0]  # Extract the audio file part before '_i'
        groups[audio_file].append(idx)  # Map audio file to its indices
    return groups

# Split audio files into train and test sets
def split_by_audio_file(groups, test_size=0.2, random_state=42):
    audio_files = list(groups.keys())
    train_files, test_files = train_test_split(audio_files, test_size=test_size, random_state=random_state)
    train_indices = [idx for file in train_files for idx in groups[file]]
    test_indices = [idx for file in test_files for idx in groups[file]]
    return train_indices, test_indices

# Main logic
if __name__ == "__main__":
    print("Defining Dataset")

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = CustomImageDataset(annotations_file, img_dir, transform=transform)
    groups = get_audio_file_groups(annotations_file)
    train_idx, test_idx = split_by_audio_file(groups, test_size=0.2)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    class_weights = calculate_class_weights(train_dataset)
    print(f"Class Weights: {class_weights}")

    model = create_model()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start = datetime.datetime.now()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        val_loss, accuracy, f1_score, cm = validate(test_dataloader, model, loss_fn)
        scheduler.step()
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}")

        # Save the model after each epoch
        torch.save(model.state_dict(), f"Models/Final/model_epoch_{epoch + 1}.pth")

    train_loss, train_accuracy, train_f1, train_cm = validate(train_dataloader, model, loss_fn)
    test_loss, test_accuracy, test_f1, test_cm = validate(test_dataloader, model, loss_fn)

    save_results("Train_Results_under95_final.txt", train_accuracy, train_loss, train_f1, train_cm)
    save_results("Test_Results_under95_final.txt", test_accuracy, test_loss, test_f1, test_cm)

    print("Done!")