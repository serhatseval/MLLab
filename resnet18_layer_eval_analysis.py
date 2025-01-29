import os
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms, models


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def create_model():
    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, features)
    )
    return model.to(device)


def load_model(model, modelPath):
    model.load_state_dict(torch.load(modelPath, map_location=device))

    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4.register_forward_hook(get_activation('layer4'))


def eval_image(model, imgeFullPath):
    image = read_image(imgeFullPath, mode=ImageReadMode.GRAY)
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


# layer 1
def draw_layer1(modelName, imageName):
    global_max = max(matrix.max() for matrix in activation['layer1'][0])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=global_max)

    fig, axs = plt.subplots(6, 11, figsize=(22, 30))
    fig.set_facecolor('xkcd:dark gray')
    plt.gcf().set_size_inches(22, 30)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.03, hspace=0.01)

    for i in range(0, 66):
        axs[int(i/11), i % 11].axis('off')
        axs[int(i / 11), i % 11].margins(0, 0)
        axs[int(i / 11), i % 11].set_axis_off()
        if i < 64:
            axs[int(i/11), i % 11].imshow(activation['layer1'][0][i].numpy(force=True), cmap='seismic', norm=norm)

    # fig.show()
    fig.savefig("layerPlots\\layer1\\"+modelName+"_"+imageName+".png")
    plt.close(fig)

# layer 2
def draw_layer2(modelName, imageName):
    global_max = max(matrix.max() for matrix in activation['layer2'][0])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=global_max)

    fig, axs = plt.subplots(8, 16, figsize=(22, 30))
    fig.set_facecolor('xkcd:dark gray')
    plt.gcf().set_size_inches(22, 30)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=-0.4)

    for i in range(0, 128):
        axs[int(i/16), i % 16].axis('off')
        axs[int(i/16), i % 16].margins(0, 0)
        axs[int(i/16), i % 16].set_axis_off()
        if i < 128:
            axs[int(i/16), i % 16].imshow(activation['layer2'][0][i].numpy(force=True), cmap='seismic', norm=norm)

    # fig.show()
    fig.savefig("layerPlots\\layer2\\"+modelName+"_"+imageName+".png")
    plt.close(fig)

# layer 3

def draw_layer3(modelName, imageName):
    global_max = max(matrix.max() for matrix in activation['layer3'][0])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=global_max)

    fig, axs = plt.subplots(12, 22, figsize=(22, 30))
    fig.set_facecolor('xkcd:dark gray')
    plt.gcf().set_size_inches(22, 30)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.01)

    for i in range(0, 264):
        axs[int(i/22), i % 22].axis('off')
        axs[int(i/22), i % 22].margins(0, 0)
        axs[int(i/22), i % 22].set_axis_off()
        if i < 256:
            axs[int(i/22), i % 22].imshow(activation['layer3'][0][i].numpy(force=True), cmap='seismic', norm=norm)

    # fig.show()
    fig.savefig("layerPlots\\layer3\\"+modelName+"_"+imageName+".png")
    plt.close(fig)

# layer 4

def draw_layer4(modelName, imageName):
    global_max = max(matrix.max() for matrix in activation['layer4'][0])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=global_max)

    fig, axs = plt.subplots(16, 32, figsize=(22, 30))
    fig.set_facecolor('xkcd:dark gray')
    plt.gcf().set_size_inches(22, 30)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.06, hspace=-0.4)

    for i in range(0, 512):
        axs[int(i/32), i % 32].axis('off')
        axs[int(i/32), i % 32].margins(0, 0)
        axs[int(i/32), i % 32].set_axis_off()
        if i < 512:
            axs[int(i/32), i % 32].imshow(activation['layer4'][0][i].numpy(force=True), cmap='seismic', norm=norm)

    # fig.show()
    fig.savefig("layerPlots\\layer4\\"+modelName+"_"+imageName+".png")
    plt.close(fig)


if __name__ == '__main__':
    modelFullPath = "Models\\Time Mask 20%\\model_timemask.pth"
    modelName = "TimeMask20"
    imagePath = "MEL_Spectograms_TimeMasking_20\\images"
    # imageName = "m3_script1_ipad_balcony1.wav_i15.png"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activation = {}
    img_height = 1000
    img_width = 400  # Length for 3 seconds of audio
    features = 2

    # imageFullPath = os.path.join(imagePath, imageName)

    model = create_model()
    load_model(model, modelFullPath)

    persons = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
    scripts = ["2"]
    recorders = ["ipad_office2"]
    parts = ["7", "10"]
    images = ""
    for person in persons:
        for script in scripts:
            for recorder in recorders:
                for part in parts:
                    imageName = person+"_script"+script+"_"+recorder+".wav_i"+part+".png"
                    imageFullPath = os.path.join(imagePath, imageName)

                    eval_image(model, imageFullPath)

                    draw_layer1(modelName, imageName)
                    draw_layer2(modelName, imageName)
                    draw_layer3(modelName, imageName)
                    draw_layer4(modelName, imageName)
