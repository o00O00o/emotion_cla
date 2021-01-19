import torch
import argparse
import pandas as pd
from tqdm import tqdm
from data_process import TextDataset, Vocabulary
from torch.utils.data import DataLoader
from utils import PadCollate
from model import RNN_Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--pad_size', default=32, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--root_dir', default="/Users/gaoyibo/Datasets/ml2020spring-hw4/", type=str)
    parser.add_argument('--embedding_dim', default=300, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--isTrain', default=False, type=bool)
    parser.add_argument('--dropout', default=0.5, type=float)
    return parser.parse_args()


def main(args):

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocabulary(args.root_dir)
    vocab.build_vocabulary()
    args.dict_size = len(vocab)

    test_dataset = TextDataset(vocab, args.root_dir, args.isTrain)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=PadCollate(args.batch_size, args.isTrain))

    model = RNN_Model(args).to(args.device)
    checkpoint = torch.load("./my_checkpoint.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    labels = []
    for idx, texts in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
        output = model(texts)
        predict = torch.gt(output, 0.5).type(torch.int16).view(-1, ).numpy().tolist()
        labels.extend(predict)
    
    idx = [num for num in range(len(labels))]

    df = pd.DataFrame({'id':idx, 'label':labels})
    df.to_csv('./predict.csv', index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
