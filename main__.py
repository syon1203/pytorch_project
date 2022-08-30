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

def main():
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='인자값을 입력합니다.')

    # dataset, model, batch size, epoch, learning rate
    parser.add_argument('--train', '-tr', required=False, default='./cifar10/train', help='Root of Trainset')
    parser.add_argument('--test', '-ts', required=False, default='./cifar10/test', help='Root of Testset')
    parser.add_argument('--model', '-m', required=False, default='resnet34', help='Name of Model')
    parser.add_argument('--batch', '-b', required=False, default=32, help='Batch Size')
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

    # transf_train = tr.Compose([tr.RandomCrop(32, padding=4), tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transf_test = tr.Compose([tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = CIFAR10(test_root, transform=transf_test)
    trainset = CIFAR10(train_root, transform=transf_test)

    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)

    # print(len(testset))
    # print(len(trainset))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3)

    for epoch in range(args.epoch):
        #print(epoch)
        train_loss, train_accuracy = training(model, trainloader, trainset, criterion, optimizer, scheduler)

        test_loss, test_accuracy = evaluation(model, testloader, testset)

        print(f'Epoch:{epoch} Train loss:{train_loss} Train accuracy:{100*train_accuracy:.4f}% Test loss:{test_loss} Test accuracy:{100*test_accuracy:.4f}%')

        is_best = test_accuracy <= best_err
        best_err = min(test_accuracy, best_err)

        if best_err == test_accuracy:
            torch.save(model.state_dict(), 'model_weights.pth')

    model.load_state_dict(torch.load('model_weights.pth'))


if __name__ == "__main__":
    main()



