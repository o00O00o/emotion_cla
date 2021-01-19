import torch
from tqdm import tqdm
from sklearn import metrics


def train(args, model, dataloader, criterion, optimizer):
    model.train()
    for idx, (text, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        text = text.to(args.device)
        label = label.to(args.device)
        output = model(text)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    true = label.data.cpu()
    predict = torch.gt(output.data, 0.5)
    train_acc = metrics.accuracy_score(true, predict)

    print("Train Loss:  {:.3f}, Train Acc:  {:.3f}".format(loss.detach().numpy(), train_acc))


def validate(args, model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    for idx, (text, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        text = text.to(args.device)
        label = label.to(args.device)
        output = model(text)
        loss = criterion(output, label)
        total_loss += loss
        true = label.data.cpu()
        predict = torch.gt(output.data, 0.5)
        acc = metrics.accuracy_score(true, predict)
        total_acc += acc
    print("Val Loss:  {:.3f}, Val Acc:  {:.3f}".format(total_loss/len(dataloader), total_acc/len(dataloader)))
