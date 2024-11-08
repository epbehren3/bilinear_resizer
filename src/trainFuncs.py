#WIP, need to edit training and testing functions to work with model.
import libraryImport as li

#Data Trainer - Training process for my resnet model
def train(model: nn.Module,
          lossFN: nn.modules.loss._Loss,
          optimizer: torch.optim.Optimizer,
          trainLoader: torch.utils.data.DataLoader,
          epoch: int=0)-> List:
    # ----------- <Your code> ---------------
    model.train()
    train_loss = []


    for batch_idx, (images, targets) in enumerate(trainLoader):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = lossFN(output, targets)
        loss.backward()
        optimizer.step()


        train_loss.append(loss.item())

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}: [{batch_idx*len(images)}/{len(trainLoader.dataset)}] Loss: {loss.item():.3f}')






    assert len(train_loss) == len(trainLoader)
    return train_loss

def test(model: nn.Module,
         lossFN: nn.modules.loss._Loss,
         testLoader: torch.utils.data.DataLoader,
         epoch: int=0)-> Dict:
  
    model.eval()

    test_stat = {
        "loss": 0,
        "accuracy": 0,
        "prediction": []
    }


    with torch.no_grad():
        for images, targets in testLoader:
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            test_stat["loss"] += lossFN(output, targets).item()
            test_stat["accuracy"] += (torch.argmax(output, dim=1) == targets).sum().item()
            test_stat["prediction"].append(torch.argmax(output, dim=1))


    test_stat["accuracy"] /= len(testLoader.dataset)
    test_stat["loss"] /= len(testLoader)
    test_stat["prediction"] = torch.cat(test_stat["prediction"])


    # ----------- <Your code> ---------------
    # dictionary should include loss, accuracy and prediction
    print(f"Accuracy: {test_stat['accuracy']:.4f}%")

    assert "loss" and "accuracy" and "prediction" in test_stat.keys()
    # "prediction" value should be a 1D tensor
    assert len(test_stat["prediction"]) == len(testLoader.dataset)
    assert isinstance(test_stat["prediction"], torch.Tensor)
    return test_stat