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
    parser.add_argument('--batch', '-b', required=False, default=32, help='Batch Size')
    parser.add_argument('--epoch', '-e', required=False, default=50, help='Epoch')
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

    transf_train = tr.Compose([tr.ToTensor(), tr.RandomCrop(32, padding=4, padding_mode='reflect'),
                               tr.RandomHorizontalFlip(), jittering,
                               lighting, tr.Normalize(*stats, inplace=True)])
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

        # 현재까지 학습한 것 중 best_err인 경우
        if best_err == test_accuracy:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, 'checkpoint.tar')



    model.load_state_dict(torch.load('model_weights.pth'))


if __name__ == "__main__":
    main()

"""
Epoch:0 Train loss:1.912684679031372 Train accuracy:27.3040% Test loss:1.7414369583129883 Test accuracy:35.0300%
Epoch:1 Train loss:1.6020643711090088 Train accuracy:40.7080% Test loss:1.4354122877120972 Test accuracy:46.0600%
Epoch:2 Train loss:1.4257917404174805 Train accuracy:47.5820% Test loss:1.219020128250122 Test accuracy:54.9700%
Epoch:3 Train loss:1.2919151782989502 Train accuracy:53.1320% Test loss:1.1193878650665283 Test accuracy:58.9900%
Epoch:4 Train loss:1.1919453144073486 Train accuracy:57.1500% Test loss:1.0136706829071045 Test accuracy:63.1700%
Epoch:5 Train loss:1.1052721738815308 Train accuracy:60.2900% Test loss:0.9589880108833313 Test accuracy:64.6300%
Epoch:6 Train loss:1.0362017154693604 Train accuracy:62.9200% Test loss:0.8881317377090454 Test accuracy:67.8000%
Epoch:7 Train loss:0.9836637377738953 Train accuracy:64.9600% Test loss:0.8830289244651794 Test accuracy:67.5600%
Epoch:8 Train loss:0.9367906451225281 Train accuracy:66.5100% Test loss:0.8109889626502991 Test accuracy:71.1600%
Epoch:9 Train loss:0.8950363993644714 Train accuracy:68.2720% Test loss:0.8063583374023438 Test accuracy:71.5800%
Epoch:10 Train loss:0.8570759892463684 Train accuracy:69.6200% Test loss:0.7831292748451233 Test accuracy:72.6400%
Epoch:11 Train loss:0.8231267929077148 Train accuracy:70.9160% Test loss:0.7076497077941895 Test accuracy:74.9700%
Epoch:12 Train loss:0.7901850938796997 Train accuracy:72.2780% Test loss:0.7179893851280212 Test accuracy:74.6800%
Epoch:13 Train loss:0.7635319828987122 Train accuracy:73.1040% Test loss:0.6586726903915405 Test accuracy:76.9600%
Epoch:14 Train loss:0.736615002155304 Train accuracy:74.2820% Test loss:0.6490936279296875 Test accuracy:77.4300%
Epoch:15 Train loss:0.7069482803344727 Train accuracy:75.1880% Test loss:0.6610475778579712 Test accuracy:77.0900%
Epoch:16 Train loss:0.6603788137435913 Train accuracy:76.9840% Test loss:0.5874813199043274 Test accuracy:79.6500%
Epoch:17 Train loss:0.6469254493713379 Train accuracy:77.4420% Test loss:0.5836829543113708 Test accuracy:79.9900%
Epoch:18 Train loss:0.6338933110237122 Train accuracy:77.8580% Test loss:0.5701333284378052 Test accuracy:80.3700%
Epoch:19 Train loss:0.6248329877853394 Train accuracy:78.2200% Test loss:0.561042070388794 Test accuracy:80.6700%
Epoch:20 Train loss:0.6131470203399658 Train accuracy:78.4140% Test loss:0.5568347573280334 Test accuracy:80.8400%
Epoch:21 Train loss:0.6050633788108826 Train accuracy:78.7800% Test loss:0.5424526333808899 Test accuracy:81.0700%
Epoch:22 Train loss:0.5818809866905212 Train accuracy:79.7620% Test loss:0.5311347246170044 Test accuracy:81.6300%
Epoch:23 Train loss:0.5698559284210205 Train accuracy:80.3080% Test loss:0.535713791847229 Test accuracy:81.5100%
Epoch:24 Train loss:0.5689359903335571 Train accuracy:80.1280% Test loss:0.5253891944885254 Test accuracy:81.9500%
Epoch:25 Train loss:0.5588540434837341 Train accuracy:80.5580% Test loss:0.531177282333374 Test accuracy:81.7100%
Epoch:26 Train loss:0.5551828742027283 Train accuracy:80.6800% Test loss:0.5206995010375977 Test accuracy:81.9800%
Epoch:27 Train loss:0.5586897730827332 Train accuracy:80.6800% Test loss:0.514803946018219 Test accuracy:82.2200%
Epoch:28 Train loss:0.5455916523933411 Train accuracy:81.0720% Test loss:0.5120292901992798 Test accuracy:82.2600%
Epoch:29 Train loss:0.5389382839202881 Train accuracy:81.2580% Test loss:0.508057713508606 Test accuracy:82.4900%
Epoch:30 Train loss:0.5397780537605286 Train accuracy:81.3000% Test loss:0.5201069116592407 Test accuracy:82.3500%
Epoch:31 Train loss:0.5385211110115051 Train accuracy:81.3400% Test loss:0.5097538828849792 Test accuracy:82.7100%
Epoch:32 Train loss:0.5269231200218201 Train accuracy:81.6040% Test loss:0.5026624202728271 Test accuracy:82.6700%
Epoch:33 Train loss:0.5320501327514648 Train accuracy:81.4600% Test loss:0.5047408938407898 Test accuracy:82.7600%
Epoch:34 Train loss:0.5258265137672424 Train accuracy:81.5560% Test loss:0.5008493661880493 Test accuracy:82.8300%
Epoch:35 Train loss:0.5258172750473022 Train accuracy:81.9960% Test loss:0.5031132698059082 Test accuracy:82.6000%
Epoch:36 Train loss:0.5244211554527283 Train accuracy:81.6320% Test loss:0.5001206398010254 Test accuracy:82.7700%
Epoch:37 Train loss:0.5207294821739197 Train accuracy:81.9420% Test loss:0.5007598400115967 Test accuracy:82.6200%
Epoch:38 Train loss:0.5181577205657959 Train accuracy:81.7900% Test loss:0.49983203411102295 Test accuracy:82.8300%
Epoch:39 Train loss:0.5187380313873291 Train accuracy:81.9540% Test loss:0.49561557173728943 Test accuracy:83.0300%
Epoch:40 Train loss:0.5200704336166382 Train accuracy:81.9560% Test loss:0.5008311867713928 Test accuracy:82.8500%
Epoch:41 Train loss:0.5197455883026123 Train accuracy:81.8800% Test loss:0.4953133761882782 Test accuracy:83.1200%
Epoch:42 Train loss:0.513386070728302 Train accuracy:82.1340% Test loss:0.507339596748352 Test accuracy:82.7100%
Epoch:43 Train loss:0.515157163143158 Train accuracy:82.0300% Test loss:0.4935731589794159 Test accuracy:83.2300%
Epoch:44 Train loss:0.5180171728134155 Train accuracy:81.9540% Test loss:0.49656152725219727 Test accuracy:83.1100%
Epoch:45 Train loss:0.5172951817512512 Train accuracy:82.0600% Test loss:0.4991973638534546 Test accuracy:82.9600%
Epoch:46 Train loss:0.51298588514328 Train accuracy:82.0800% Test loss:0.49403905868530273 Test accuracy:82.8500%
Epoch:47 Train loss:0.5145379900932312 Train accuracy:82.0320% Test loss:0.49921709299087524 Test accuracy:82.8900%
Epoch:48 Train loss:0.511850118637085 Train accuracy:82.1700% Test loss:0.49938175082206726 Test accuracy:82.8700%
Epoch:49 Train loss:0.517962634563446 Train accuracy:82.0060% Test loss:0.49690020084381104 Test accuracy:83.0500%
"""


