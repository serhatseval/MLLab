import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms, models
import os

img_height = 1025
img_width = 259  # Length for 3 seconds of audio
features = 2

# Define model
def create_model():
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single-channel input
    model.fc = nn.Linear(model.fc.in_features, features)  # Adjust the final layer for your task
    return model.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model()

# Load the model weights
try:
    model.load_state_dict(torch.load("OutputFiles/model_fold1.pth", map_location=device))
except RuntimeError as e:
    print(f"Error loading model: {e}")
    # Optionally, try partial load
    state_dict = torch.load("OutputFiles/model_fold1.pth", map_location=device)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})

model.eval()

def predict_image(image_path):
    image = read_image(image_path, mode=ImageReadMode.GRAY)
    
    # Convert the image to float and normalize
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),  # Reshape to expected input size
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class