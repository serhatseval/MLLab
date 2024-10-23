import torch
from torchvision.io import read_image, ImageReadMode
from CNN import NeuralNetwork

classes = [
    "Foe",
    "Friend"
]

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")


def load_model(model_path):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()
    return model


def decide(image_path, model):
    x = read_image(image_path, mode=ImageReadMode.GRAY)
    with torch.no_grad():
        x = x.to(device)
        prediction = model(x)
    return prediction[0].argmax[0]
