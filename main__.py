import util
import model2 as MD
from train import training
import torchvision.transforms as tr
from torch.utils.data import DataLoader
from dataset import CIFAR10
from test import evaluation
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def main():
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='인자값을 입력합니다.')

    # dataset, model, batch size, epoch, learning rate
    parser.add_argument('--train', '-tr', required=False, default='./cifar10/train', help='Root of Trainset')
    parser.add_argument('--test', '-ts', required=False, default='./cifar10/test', help='Root of Testset')
    parser.add_argument('--model', '-m', required=False, default='resnet34', help='Name of Model')
    parser.add_argument('--batch', '-b', required=False, default=4, help='Batch Size')
    parser.add_argument('--epoch', '-e', required=False, default=10, help='Epoch')
    parser.add_argument('--lr', '-l', required=False, default=0.001, help='Learning Rate')

    best_err = 100

    # 입력받은 인자값을 args에 저장
    args = parser.parse_args()

    # 입력받은 인자값 출력
    print(args.train)
    print(args.test)
    print(args.model)
    print(args.batch)
    print(args.epoch)
    print(args.lr)

    train_root = args.train
    test_root = args.test

    # model
    if args.model == 'resnet34':
        model = MD.ResNet34('cifar10')
    elif args.model == 'resnet18':
        model = MD.ResNet18('cifar10')
    else:
        model = None

    # Data transforms (normalization & data augmentation)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    jittering = util.ColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.4)
    lighting = util.Lighting(alphastd=0.1,
                              eigval=[0.2175, 0.0188, 0.0045],
                              eigvec=[[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]])

    transf_train = tr.Compose([tr.ToTensor(), tr.RandomCrop(32, padding=4), tr.RandomHorizontalFlip(), jittering, lighting, tr.Normalize(*stats, inplace=True)])
    transf_test = tr.Compose([tr.ToTensor(), tr.Normalize(*stats, inplace=True)])

    testset = CIFAR10(test_root, transform=transf_test)
    trainset = CIFAR10(train_root, transform=transf_train)

    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)

    def show_batch(dl):
        for images, labels in dl:
            # image = (image/np.amax(image) + 1)
            # image = image/np.amax(image)
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_xticks([]);
            ax.set_yticks([])
            ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0).numpy())
            print(labels)
            #plt.show()
            break

    #show_batch(trainloader)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3)

    for epoch in range(args.epoch):

        train_loss, train_accuracy = training(model, trainloader, criterion, optimizer, scheduler)

        test_loss, test_accuracy = evaluation(model, testloader, criterion)

        print(f'Epoch:{epoch} Train loss:{train_loss} Train accuracy:{100*train_accuracy:.4f}% Test loss:{test_loss} Test accuracy:{100*test_accuracy:.4f}%')

        best_err = test_accuracy <= best_err
        best_err = min(test_accuracy, best_err)

        if best_err == test_accuracy:
            torch.save(model.state_dict(), 'model_weights.pth')

    model.load_state_dict(torch.load('model_weights.pth'))


if __name__ == "__main__":
    main()

"""
Epoch:0 Train loss:1.8830300569534302 Train accuracy:30.4620% Test loss:1.4114612340927124 Test accuracy:48.7400%
Epoch:1 Train loss:1.3994910717010498 Train accuracy:49.8900% Test loss:1.117986798286438 Test accuracy:61.2300%
Epoch:2 Train loss:1.10672926902771 Train accuracy:60.8920% Test loss:0.9473481178283691 Test accuracy:67.3300%
Epoch:3 Train loss:0.9372214674949646 Train accuracy:67.3540% Test loss:0.7944387793540955 Test accuracy:72.8700%
Epoch:4 Train loss:0.8333507776260376 Train accuracy:70.9560% Test loss:0.7847883105278015 Test accuracy:74.1100%
Epoch:5 Train loss:0.7588878273963928 Train accuracy:73.8080% Test loss:0.7335292100906372 Test accuracy:74.5800%
Epoch:6 Train loss:0.6561610102653503 Train accuracy:77.4260% Test loss:0.6177797317504883 Test accuracy:79.0200%
Epoch:7 Train loss:0.6217713952064514 Train accuracy:78.5260% Test loss:0.5766865015029907 Test accuracy:80.1500%
Epoch:8 Train loss:0.597963273525238 Train accuracy:79.3080% Test loss:0.5386070013046265 Test accuracy:81.9400%
Epoch:9 Train loss:0.5734032392501831 Train accuracy:80.1240% Test loss:0.5370789766311646 Test accuracy:81.7500%
"""


