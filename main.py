import torch
import argparse
import numpy as np
from learning import train, validate
import torch.nn as nn
import torch.optim as optim
from data_process import TextDataset, Vocabulary
from torch.utils.data import DataLoader, random_split
from utils import PadCollate
from model import RNN_Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--pad_size', default=32, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--root_dir', default="/Users/gaoyibo/Datasets/ml2020spring-hw4/", type=str)
    parser.add_argument('--embedding_dim', default=300, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--isTrain', default=True, type=bool)
    parser.add_argument('--train_size', default=0.01, type=float)
    return parser.parse_args()


def main(args):

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocabulary(args.root_dir)
    vocab.build_vocabulary()
    word_embedding_weight = torch.from_numpy(np.load('./word_embedding_weight.npy'))

    text_dataset = TextDataset(vocab, args.root_dir, args.isTrain)

    train_size = int(args.train_size * len(text_dataset))
    test_size = len(text_dataset) - train_size

    train_dataset, val_dataset = random_split(text_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=PadCollate(args.batch_size, args.isTrain))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=PadCollate(args.batch_size, args.isTrain))

    model = RNN_Model(args, word_embedding_weight).to(args.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        train(args, model, train_loader, criterion, optimizer)
        validate(args, model, val_loader, criterion)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, 'my_checkpoint.pth.tar')


if __name__ == "__main__":
    args = parse_args()
    main(args)
