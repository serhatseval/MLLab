import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import os
import random
from torchvision.io import read_image, ImageReadMode


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def create_model():
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)  # Assuming 2 classes
    )
    return model.to(device)

# Prune the model
def prune_model(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

# Custom Dataset (assuming you have a dataset class)
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to evaluate the model
def evaluate_model(dataloader, model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm

def main():
    # Load the model
    model = create_model()
    
    # Load the pre-trained model weights
    model_path = 'Models/Eliminating Files over 95% Similarity Epoch Saving for Comparison With Pruning/model_epoch_10.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Apply pruning
    pruned_model = prune_model(model, amount=0.5)
    
    # Define the dataset and dataloader
    annotations_file = 'MEL_Spectograms/labels_removing_95_percent_similarity.csv'  
    img_dir = "MEL_Spectograms/images"
    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    dataset = CustomImageDataset(annotations_file, img_dir, transform=transform)
    
    # Randomly select 1000 spectrograms for testing
    if len(dataset) > 10000:
        indices = random.sample(range(len(dataset)), 10000)
        subset = Subset(dataset, indices)
    else:
        subset = dataset
    
    dataloader = DataLoader(subset, batch_size=16, shuffle=False)
    
    # Evaluate the pruned model
    accuracy, cm = evaluate_model(dataloader, pruned_model)
    
    # Save the results
    with open('pruned_model_results.txt', 'w') as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write("Confusion Matrix:\n")
        file.write(f"{cm}\n")

if __name__ == "__main__":
    main()