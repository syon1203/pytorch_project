import torch

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def evaluation(model, dataloader, criterion):
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_accuracy = 0.0
        data_len = 0

        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.cpu().data * images.size(0)  # loss.cpu().data = loss
            _, prediction = torch.max(outputs.data, 1)

            test_accuracy += int(torch.sum(prediction == labels.data))
            data_len += len(prediction)

        test_loss = test_loss / data_len
        test_accuracy = test_accuracy / data_len

    return test_loss, test_accuracy

