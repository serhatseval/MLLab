import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models

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

# Function to print the weights of the model
def print_weights(model, file):
    for name, param in model.named_parameters():
        if 'weight' in name:
            file.write(f"Weights of layer {name}:\n")
            file.write(f"{param.data}\n")

def main():
    model = create_model()
    
    model_path = 'Models/Eliminating Files over 95% Similarity Epoch Saving for Comparison With Pruning/model_epoch_10.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    with open('pruning_results.txt', 'w') as file:
        file.write("Weights before pruning:\n")
        print_weights(model, file)
        
        pruned_model = prune_model(model, amount=0.5)
        
        file.write("\nWeights after pruning:\n")
        print_weights(pruned_model, file)

if __name__ == "__main__":
    main()