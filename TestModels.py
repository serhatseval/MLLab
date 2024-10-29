import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the annotation files and image directories
test_annotations_file = 'OutputFilesSeperated/Clean/labels.csv'
test_img_dir = 'OutputFilesSeperated/Clean/images'
model_dir = 'OutputFiles'  # Directory where models are saved
results_dir = 'Models/TestResults'  # Directory to save the results

img_height = 1025
img_width = 259  # Length for 3 seconds of audio
features = 2  # Number of classes
batch_size = 16

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
        try:
            image = read_image(img_path, mode=ImageReadMode.GRAY)
        except FileNotFoundError:
            print(f"File not found: {img_path}. Skipping.")
            return None, None
        label = int(self.img_labels.iloc[idx, 1])  # Ensure label is integer
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Get cpu, gpu or mps device for testing.
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

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Allowed', 'Not Allowed'], yticklabels=['Allowed', 'Not Allowed'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(results_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            if X is None or y is None:
                continue
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_loss /= num_batches
    correct /= size
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    return correct, test_loss, f1, cm

if __name__ == "__main__":
    print("Defining Dataset")
    
    # Define transforms for test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_data = CustomImageDataset(test_annotations_file, test_img_dir, transform=test_transform,
                                   target_transform=None)

    print("Creating DataLoader")
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()

    # Create directory for results if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Test each model
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.pth'):
            model_path = os.path.join(model_dir, model_file)
            print(f"Testing model: {model_path}")
            model = create_model()
            model.load_state_dict(torch.load(model_path, map_location=device))
            accuracy, avg_loss, f1_score, cm = test(test_dataloader, model, loss_fn)
            print(f"Model: {model_file}, Accuracy: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}, F1 Score: {f1_score:.4f}")
            print(f"Confusion Matrix:\n{cm}")
            
            # Save metrics and confusion matrix
            with open(os.path.join(results_dir, f"{model_file}_metrics.txt"), 'w') as f:
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Avg Loss: {avg_loss:.4f}\n")
                f.write(f"F1 Score: {f1_score:.4f}\n")
                f.write(f"Confusion Matrix:\n{cm}\n")
            
            plot_confusion_matrix(cm, model_file)

    print("Testing complete. Results saved to", results_dir)