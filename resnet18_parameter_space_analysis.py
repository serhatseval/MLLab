import os
import re
import numpy as np
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms, models


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


def get_state_dict(modelPath):
    return torch.load(modelPath, map_location=device)


def get_val_dicts(state_dict):
    conv_dict = dict()
    bn_w_dict = dict()
    bn_b_dict = dict()
    downsample_w_dict = dict()
    downsample_b_dict = dict()

    for key, val in state_dict.items():
        if re.match(".*conv.*", key):
            conv_dict[key] = val
        if re.match(".*bn..weight.*", key):
            bn_w_dict[key] = val
        if re.match(".*bn..bias.*", key):
            bn_b_dict[key] = val
        if re.match(".*downsample...weight.*", key):
            downsample_w_dict[key] = val
        if re.match(".*downsample...bias.*", key):
            downsample_b_dict[key] = val
    return [conv_dict, bn_w_dict, bn_b_dict, downsample_w_dict, downsample_b_dict]


def dict_elementwise_diff(dict1, dict2):
    sum_abs = 0
    sum_rel = 0
    for key, val in dict1.items():
        temp1 = torch.sub(dict1[key], dict2[key])
        temp2 = torch.div(temp1, dict1[key])
        sum_abs = sum_abs + abs(torch.sum(temp1))
        sum_rel = sum_rel + abs(torch.sum(temp2))
    return sum_abs.item(), sum_rel.item()


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

    sd = get_state_dict(modelFullPath)

    partialPath = "Models\\pruned\\model_epoch_"

    sd2 = get_state_dict(partialPath + "1.pth")
    vd2 = get_val_dicts(sd2)

    arr_abs = np.zeros([5, 9])
    arr_rel = np.zeros([5, 9])
    text = [0]*9
    for n in range(2, 11):
        sd1 = sd2
        vd1 = vd2
        sd2 = get_state_dict(partialPath + str(n) + ".pth")
        vd2 = get_val_dicts(sd2)
        text[n-2] = n
        for i in range(5):
            temp = dict_elementwise_diff(vd1[i], vd2[i])
            arr_abs[i][n-2] = temp[0]
            arr_rel[i][n-2] = temp[1]

    print(arr_abs)
    print(arr_rel)

    plt.plot(text, arr_abs.transpose())
    plt.legend(["conv weights", "norm weights", "norm biases", "downsample weight", "downsample biases"], bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.yscale('log')
    plt.tight_layout()
    plt.show()