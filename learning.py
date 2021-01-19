import torch
from tqdm import tqdm
from sklearn import metrics


def train(args, model, dataloader, criterion, optimizer):
    model.train()
    for epoch in range(args.epoch):
        for idx, (text, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
            text = text.to(args.device)
            label = label.to(args.device)
            output = model(text)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 1000 == 0:
                true = label.data.cpu()
                predict = torch.gt(output.data, 0.5)
                train_acc = metrics.accuracy_score(true, predict)
                print(train_acc)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, 'my_checkpoint.pth.tar')