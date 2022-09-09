import torch


def evaluation(model, dataloader, criterion, device):
    model.to(device)
    model.eval() #eval 로 바꾸고

    with torch.no_grad(): #메모리 사용량 줄이기 위해 gradient 계산 x , 연산속도 증가
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

