import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def ResNet34():
    resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    resnet.aux_logits = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)  # output 크기 10
    resnet.to(device)

    return resnet

def ResNet18():
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.aux_logits = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)  # output 크기 10
    resnet.to(device)

    return resnet
