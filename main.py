import torch
import argparse
from learning import train
import torch.nn as nn
import torch.optim as optim
from data_process import TextDataset, Vocabulary
from torch.utils.data import DataLoader
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
    return parser.parse_args()


def main(args):

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocabulary(args.root_dir)
    vocab.build_vocabulary()
    word_embedding_weight = torch.load('./word_embedding_weight.pth')

    train_dataset = TextDataset(vocab, args.root_dir, args.isTrain)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=PadCollate(args.batch_size, args.isTrain))

    model = RNN_Model(args, word_embedding_weight).to(args.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, train_loader, criterion, optimizer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
