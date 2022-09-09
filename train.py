import torch


def training(model, dataloader, criterion,  optimizer, scheduler, device):
    model.to(device)
    model.train()

    train_accuracy = 0.0
    train_loss = 0.0
    data_len = 0

      # Training
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Making the gradients 0 at the start of a new batch 그래디언트 0으로 만들기

        outputs = model(images) #모델에 이미지 넣기
        loss = criterion(outputs, labels) #로스 계산
        loss.backward()  # Backpropagation
        optimizer.step() #bachpropagation에서 이루어진 변화도를 통해 매개변수 조정, 배치마다 해줌

        train_loss += loss.cpu().data * images.size(0)  # loss.cpu().data = loss
        _, prediction = torch.max(outputs.data, 1) #텐서의 최대값
        """_ 값의 의미
        
        outputs.data의 크기 == batch size * class 개수
        torch.max(outputs.data, 1)에서 각 열은 하나의 이미지를 나타내는 벡터이므로 열 기준으로 최대값을 뽑아 예측값을 만든다는 의미
        
        예시)
                torch.return_types.max(
                values=tensor([0.9280, 0.4403, 0.7147, 1.1411, 0.9623, 0.6701, 0.5891, 1.2484, 0.9073,
                0.8828, 0.3879, 0.5823, 0.4257, 0.5665, 0.6186, 0.4464, 0.5019, 0.6142,
                0.8364, 0.5403, 1.2740, 1.1892, 0.6313, 0.6219, 0.5843, 1.2104, 0.5588,
                0.7127, 0.4161, 1.0917, 0.6308, 0.5997]),
                indices=tensor([2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2]))
        
        torch.max는 최대값, 최대값의 위치를 리턴해줌
        최대값 자체는 필요가 없었으므로 최대값의 위치만 prediction 에서 사용한다.
        
        따라서 torch.max(outputs.data, 1)는 최대값의 위치를 열마다(하나의 이미지마다)받아 예측값으로 사용한다.
        
        + 앞서 with torch.no_grad(): 를 사용하지 않았기 때문에 outputs *.data* 를 통해 역전파를 사용하지 않고 데이터만 사용하도록 작성하였다. 
        
        """

        train_accuracy += int(torch.sum(prediction == labels.data))
        data_len += len(prediction)

        # Print loss, accuracy of every 2000 batch
        if i % 2000 == 1999:
            batch_loss = train_loss / data_len
            batch_accuracy = train_accuracy / data_len

            print(f'batch {i // len(prediction)} : train loss {batch_loss} , '
                  f'train accuracy {100*batch_accuracy:.4f}%')
    # Sum of len(mini batch) == data_len
    train_loss = train_loss / data_len
    train_accuracy = train_accuracy / data_len
    scheduler.step(loss)  # Update the weight and bias, epoch 마다 해줌

    return train_loss, train_accuracy

