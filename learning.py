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
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                print(train_acc)
