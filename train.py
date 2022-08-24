import torch.nn as nn
import torch.optim as optim
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau

#model = nn.DataParallel(model)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


#sample = next(iter(trainloader))
#print(type(sample))
#print(sample)
#print(sample.size())


def training(model, dataloader, dataset, learning_rate=0.001, total_epochs=10):


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3)


    epochs = total_epochs # 전체 데이터를 활용한 학습 진행 횟수

    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
      # Training
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Making the gradients 0 at the start of a new batch

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            scheduler.step(loss)  # Update the weight and bias

            train_loss += loss.cpu().data * images.size(0)  # loss.cpu().data = loss
            _, prediction = torch.max(outputs.data, 1)

            train_accuracy += int(torch.sum(prediction == labels.data))

        train_loss = train_loss / len(dataset)
        train_accuracy = train_accuracy / len(dataset)

    return train_loss, train_accuracy

