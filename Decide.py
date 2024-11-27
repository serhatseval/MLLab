import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms, models
import os

img_height = 1000
img_width = 400  # Length for 3 seconds of audio
features = 2

our_model = 'Models/Eliminating Files over 95% Similarity/model_under95.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model():
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, features)
    )
    return model.to(device)

# Load model
model = create_model()
try:
    model.load_state_dict(torch.load(our_model, map_location=device))
except RuntimeError as e:
    print(f"Error loading model: {e}")
    state_dict = torch.load(our_model, map_location=device)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
model.eval()


def predict_image(image_path):
    image = read_image(image_path, mode=ImageReadMode.GRAY)
    
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)), 
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class