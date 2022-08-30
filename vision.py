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

batch_size = 32 # minibatch size
epochs = 10 # 전체 데이터를 활용한 학습 진행 횟수

#경로
train_root = './cifar10/train'
test_root = './cifar10/test'


def get_labels(root):
    label = [f for f in os.listdir(root) if not f.startswith('.')]
    labels = {i: string for i, string in enumerate(label)}
    print(labels)
    return labels


def match_image(root):
    images = []
    labels = {}

    for i, label in enumerate(os.listdir(root)):
        labels[i] = label
        try:
            for i in os.listdir(os.path.join(root, label)):
                image = img.imread(os.path.join(root, label,i))
                images.append((label, image))
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
        #image = Image.fromarray(np.uint8(img))

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return image, label

    def __len__(self):
        return len(self.images)


#transf_train = tr.Compose([tr.RandomCrop(32, padding=4), tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
transf_test = tr.Compose([tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

testset = CIFAR10(test_root,transform=transf_test)
trainset = CIFAR10(train_root,transform=transf_test)

testloader = DataLoader(testset, batch_size=8, shuffle=False)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

#print('number of test data: ', len(testset))

from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xticks([]); ax.set_yticks([])
        print(labels)
        #ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break


show_batch(trainloader)
show_batch(testloader)


resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10) # output 크기 10


from torch.optim.lr_scheduler import ReduceLROnPlateau

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3)
best_acc = 0.0

for epoch in range(epochs):
    # Training
    resnet.train()
    train_accuracy = 0.0
    train_loss = 0.0
    # For each batch in trainloader
    for i, (images, labels) in enumerate(trainloader):
        #images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Making the gradients 0 at the start of a new batch

        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        scheduler.step(loss)  # Update the weight and bias

        train_loss += loss.cpu().data * images.size(0)  # loss.cpu().data = loss
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_loss = train_loss / len(trainset)
    train_accuracy = train_accuracy / len(trainset)

    # evaluation 전환
    resnet.eval()
    test_loss = 0.0
    test_accuracy = 0.0

    for i, (images, labels) in enumerate(testloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        output, x = resnet(images)
        loss = criterion(output, labels)

        test_loss += loss.cpu().data * images.size(0)  # loss.cpu().data = loss
        _, prediction = torch.max(outputs.data, 1)

        test_accuracy += int(torch.sum(prediction == labels.data))

    test_loss = test_loss / len(testset)
    test_accuracy = test_accuracy / len(testset)

    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Loss: '+str(test_loss)+' Test Accuracy: '+str(100*test_accuracy) +'%')
