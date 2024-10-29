import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import KFold

# Paths to the annotation files and image directories
train_annotations_file = 'OutputFilesSeperated/CNN/labels.csv'
train_img_dir = 'OutputFilesSeperated/CNN/images'
test_annotations_file = 'OutputFilesSeperated/Clean/labels.csv'
test_img_dir = 'OutputFilesSeperated/Clean/images'

img_height = 1025
img_width = 259  # Length for 3 seconds of audio
features = 2  # Number of classes
batch_size = 16
epochs = 10

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
        label = int(self.img_labels.iloc[idx, 1])  # Ensure label is integer
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

# Define model
def create_model():
    model = models.resnet18(pretrained=True)
    
    # Modify the first conv layer to accept single-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Adjust the final layer for the number of classes
    model.fc = nn.Linear(model.fc.in_features, features)
    return model.to(device)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  " +
                  f"{(datetime.datetime.now()-start).total_seconds() // 60} min")


def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    all_preds = []
    all_labels = []
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


def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Allowed', 'Not Allowed'], yticklabels=['Allowed', 'Not Allowed'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'ConfusionMatrices/{model_name}_confusion_matrix.png')
    plt.close()


if __name__ == "__main__":
    print("Defining Dataset")
    
    # Define transforms for training and testing data
    train_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset and dataloaders
    dataset = CustomImageDataset(train_annotations_file, train_img_dir, transform=train_transform)
    print("Creating DataLoader")

    kf = KFold(n_splits=5)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model, loss function, and optimizer
        model = create_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Track training start time
        start = datetime.datetime.now()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            val_loss, accuracy, f1_score, cm = validate(val_dataloader, model, loss_fn)
            scheduler.step()
            print(f"Epoch {epoch + 1} training finished after {(datetime.datetime.now() - start).total_seconds() // 60} minutes from start")
        
        # Save the model after each fold
        model_name = f"model_fold{fold + 1}"
        torch.save(model.state_dict(), f"Models/{model_name}.pth")
        print(f"Saved model for fold {fold + 1}!")

    print("Done!")