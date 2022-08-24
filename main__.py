import model as RN
from train import training
import torchvision.transforms as tr
from torch.utils.data import DataLoader
from dataset import CIFAR10
from test import evaluation
import argparse
import torch

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description='인자값을 입력합니다.')

# dataset, model, batch size, epoch, learning rate
parser.add_argument('--data', '-d', required=False, default='cifar10', help='Name of Dataset')
parser.add_argument('--model', '-m', required=False, default='resnet34', help='Name of Model')
parser.add_argument('--batch', '-b', required=False, default=32, help='Batch Size')
parser.add_argument('--epoch', '-e', required=False, default=10, help='Epoch')
parser.add_argument('--lr', '-l', required=False, default=0.001, help='Learning Rate')

# 입력받은 인자값을 args에 저장
args = parser.parse_args()

# 입력받은 인자값 출력
print(args.data)
print(args.model)
print(args.batch)
print(args.epoch)
print(args.lr)


# root
if args.data == 'cifar10':
    train_root = './cifar10/train'
    test_root = './cifar10/test'

# model
if args.model == 'resnet34':
    model = RN.ResNet34("ResNet34_Weights.DEFAULT")
elif args.model == 'resnet18':
    model = RN.ResNet18("ResNet18_Weights.DEFAULT")
else:
    model = None

torch.save(model.state_dict(), 'model_weights.pth')

# transf_train = tr.Compose([tr.RandomCrop(32, padding=4), tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
transf_test = tr.Compose([tr.ToTensor(),tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

testset = CIFAR10(test_root,transform=transf_test)
trainset = CIFAR10(train_root,transform=transf_test)

testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)
trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)

#print(len(testset))
#print(len(trainset))

training(model, trainloader, trainset, args.lr, args.epoch)

if args.model == 'resnet34':
    model = RN.ResNet34()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

elif args.model == 'resnet18':
    model = RN.ResNet18()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
else:
    model = None

evaluation(model, testloader, testset)


