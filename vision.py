import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as tr
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights
import torch.optim as optim
import numpy as np
import matplotlib.image as img

#경로
train_root = './cifar10/train'
test_root = './cifar10/test'


def get_labels(root):
    label = [f for f in os.listdir(root) if not f.startswith('.')]
    labels = {string: i for i, string in enumerate(label)}
    return labels


def match_image(root):
    images = []
    labels = {}

    for i, label in enumerate(os.listdir(root)):
        labels[i] = label
        try:
            for i in os.listdir(os.path.join(root, label)):
                image = img.imread(os.path.join(root, label,i))
                images.append((i, image))
        except:
            pass

    print("finished")
    return images


class CIFAR10(Dataset):

    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.root = root
        self.labels = get_labels(root)
        self.images = match_image(root)
        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self.images[index][1]
        label = self.images[index][0]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}

        return sample

    def __len__(self):
        return len(self.images)


transf_train = tr.Compose([tr.RandomCrop(32, padding=4), tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
transf_test = tr.Compose([tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

testset = CIFAR10(test_root,transform=transf_test)
trainset = CIFAR10(train_root,transform=transf_train)

testloader = DataLoader(testset, batch_size=8, shuffle=False)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)


