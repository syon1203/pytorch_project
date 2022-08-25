import torch
import torch.nn as nn
from torch.autograd import Variable


def evaluation(model, dataloader, dataset):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0

    for i, (images, labels) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.cpu().data * images.size(0)  # loss.cpu().data = loss
        _, prediction = torch.max(outputs.data, 1)

        test_accuracy += int(torch.sum(prediction == labels.data))

    test_loss = test_loss / len(dataset)
    test_accuracy = test_accuracy / len(dataset)

    return test_loss, test_accuracy

