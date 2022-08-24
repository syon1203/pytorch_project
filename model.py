import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def ResNet34(weight):
    resnet = models.resnet34(weights=weight)
    resnet.aux_logits = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)  # output 크기 10
    resnet.to(device)

    return resnet

def ResNet18(weight):
    resnet = models.resnet18(weights=weight)
    resnet.aux_logits = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)  # output 크기 10
    resnet.to(device)

    return resnet
