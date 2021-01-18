import os
import re
import torch
from torch.nn.utils.rnn import pad_sequence


def get_text(root_dir, isLabeled=False, isTest=False):
    sentences = []

    if isTest:
        data_path = os.path.join(root_dir, 'testing_data.txt')
    else:
        if isLabeled:
            data_path = os.path.join(root_dir, 'training_label.txt')
        else:
            data_path = os.path.join(root_dir, 'training_nolabel.txt')

    data_file = open(data_path)
    lines = data_file.readlines()

    for line in lines:
        if isTest:
            line = line[line.find(',')+1:]
        else:
            if isLabeled:
                line = line[10:]
            else:
                line = line

        sentence = re.sub(r"[^A-Za-z’]", " ", line)
        sentence = re.sub(r" +", " ", sentence)
        sentences.append(sentence)
    data_file.close()
    return sentences

def get_corpus_text(corpus_dir):
    sentences = get_text(corpus_dir, isTest=True)
    sentences.extend(get_text(corpus_dir, isLabeled=True))
    sentences.extend(get_text(corpus_dir))
    return sentences

class PadCollate:
    def __init__(self, batch_size, pad_idx=0):
        self.pad_idx = pad_idx
        self.batch_size = batch_size

    def __call__(self, batch):
        texts = [item[0] for item in batch]
        texts = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)
        labels = [item[1] for item in batch]
        labels = torch.FloatTensor(labels).view(self.batch_size, 1)
        return texts, labels


if __name__ == "__main__":
    root_dir = "/Users/gaoyibo/Datasets/ml2020spring-hw4/"
    data_path = os.path.join(root_dir, 'testing_data.txt')
    file = open(data_path)
    lines = file.readlines()
    print(lines[18])
    line = lines[18]
    print(line[line.find(',')+1:])
    file.close()