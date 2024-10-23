import os
import pandas as pd
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import datetime


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(conv_channels_1, conv_channels_2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=max_pool_kernel_size, stride=max_pool_kernel_size, padding=0)
        self.conv2 = nn.Conv2d(conv_channels_2, conv_channels_3, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(starting_nodes_number, 128)
        self.fc2 = nn.Linear(128, features)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, starting_nodes_number)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\t{(datetime.datetime.now()-start).total_seconds()//60} min")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n" +
          f"Timestamp: {(datetime.datetime.now()-start).total_seconds()//60} minutes from start")



if __name__ == "__main__":
    annotations_file = 'OutputFiles/labels.csv'
    img_dir = './OutputFiles/images'
    img_height = 1025
    img_width = 862
    features = 2
    batch_size = 16

    print("Defining Dataset")
    training_data = CustomImageDataset(annotations_file, img_dir, transform=transforms.ConvertImageDtype(torch.float32),
                                       target_transform=None)
    test_data = CustomImageDataset(annotations_file, img_dir, transform=transforms.ConvertImageDtype(torch.float32),
                                   target_transform=None)

    print("Creating DataLoader")
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    print(f"Using {device} device")

    conv_channels_1 = 1
    conv_channels_2 = 1
    conv_channels_3 = 1
    max_pool_kernel_size = 8
    starting_nodes_number = (conv_channels_3 * (img_height // (max_pool_kernel_size * max_pool_kernel_size)) *
                             (img_width // (max_pool_kernel_size * max_pool_kernel_size)))

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    start = datetime.datetime.now()
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print(
            f"Epoch {t + 1} training finished after {(datetime.datetime.now() - start).total_seconds() // 60} minutes from start")
        test(test_dataloader, model, loss_fn)
        print(
            f"Epoch {t + 1} testing finished after {(datetime.datetime.now() - start).total_seconds() // 60} minutes from start")
        torch.save(model.state_dict(), f"OutputFiles/model{t + 1}.pth")
        print("Saved!")
    print("Done!")
