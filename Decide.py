import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import os

# --- Define the model architecture ---
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(208, 128)  # Size for my models 64
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 208)  # Size for my models 64
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("OutputFiles/model10.pth", map_location=device))  # Use your model path
model.eval() 

def predict_image(image_path):
    image = read_image(image_path, mode=ImageReadMode.GRAY)
    
    
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output).item()
    
    return predicted_class

