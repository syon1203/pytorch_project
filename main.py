import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as tr
import numpy as np
import torchvision.models as models

BATCH_SIZE = 32 # minibatch size
epochs = 5 # 전체 데이터를 활용한 학습 진행 횟수
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transf = tr.Compose([tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# CIFAR10 custom dataset
trainset = torchvision.datasets.ImageFolder(root='./train',transform=transf)
testset = torchvision.datasets.ImageFolder(root='./test',transform=transf)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# 학습용 이미지 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print(images.shape) #torch.Size([32, 3, 32, 32])
# print((torchvision.utils.make_grid(images)).shape)
# print(' '.join('%5s ' % labels[j] for j in range(32)))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# imshow(torchvision.utils.make_grid(images))
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))

resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(512, 10) # output 크기 10

imgs = 0
for n, (img, labels) in enumerate(trainloader):
    print(n,img.shape, labels.shape)
    imgs = img
    break

#from torchsummary import summary
#summary(resnet, input_size=(3, 224, 224))

resnet.eval()
out = resnet(imgs)
print(out.shape)

_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

_, index = torch.max(out, 1)
print(classes[index[0]], percentage[index[0]].item())

