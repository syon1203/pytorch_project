import torch


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def training(model, dataloader, criterion,  optimizer, scheduler):

    model.to(device)
    model.train()

    train_accuracy = 0.0
    train_loss = 0.0
    data_len = 0

      # Training
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Making the gradients 0 at the start of a new batch

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)  # loss.cpu().data = loss
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))
        data_len += len(prediction)

    #dataset 넘겨주지 않기
    train_loss = train_loss / data_len
    train_accuracy = train_accuracy / data_len
    scheduler.step(loss)  # Update the weight and bias

    return train_loss, train_accuracy

