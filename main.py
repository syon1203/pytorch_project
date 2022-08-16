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



batch_size = 32 # minibatch size
epochs = 5 # 전체 데이터를 활용한 학습 진행 횟수
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transf_train = tr.Compose([tr.RandomCrop(32, padding=4), tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
transf_test = tr.Compose([tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


# CIFAR10 custom dataset
trainset = torchvision.datasets.ImageFolder(root='./train',transform=transf_train)
testset = torchvision.datasets.ImageFolder(root='./test',transform=transf_test)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 학습용 이미지 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print(images.shape) #torch.Size([32, 3, 32, 32])
# print((torchvision.utils.make_grid(images)).shape)
# print(' '.join('%5s ' % labels[j] for j in range(32)))

resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10) # output 크기 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9)

resnet.eval()

num_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
#print(num_params)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = resnet(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
