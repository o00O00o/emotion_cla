import re
import torch
import spacy
import gensim
from utils import get_corpus_text
from torch.utils.data import Dataset


nlp = spacy.load('en')
stop_words = nlp.Defaults.stop_words

class Vocabulary:
    def __init__(self, corpus_dir, freq_threshold=5):

        self.sentences = get_corpus_text(corpus_dir)
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenize(text):
        text_tokens = [tok.text.lower() for tok in nlp.tokenizer(text)]
        tokens_without_stop_words = [word for word in text_tokens if word not in stop_words]
        return tokens_without_stop_words

    def build_vocabulary(self):
        frequencies = {}
        idx = 4

        for sentence in self.sentences:
            for word in self.tokenize(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

    def get_wv(self):
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format("~/.cache/torch/hub/checkpoints/GoogleNews-vectors-negative300.bin.gz", binary=True)
        vocab_size = len(self.stoi)
        embed_size = 300
        weight = torch.zeros(vocab_size, embed_size)

        for i in range(len(wvmodel.index2word)):
            try:
                index = self.stoi[wvmodel.index2word[i]]
            except:
                continue
            weight[index, :] = torch.from_numpy(wvmodel.get_vector(self.itos[self.stoi[wvmodel.index2word[i]]]).copy())
        return weight


class TextDataset(Dataset):
    def __init__(self, vocab, root_dir):
        self.sentences = []
        self.labels = []
        data_file = open(root_dir)
        lines = data_file.readlines()
        for line in lines:
            self.labels.append(int(line[0]))
            sentence = re.sub(r"[^A-Za-z’]", " ", line[10:])
            sentence = re.sub(r" +", " ", sentence)
            self.sentences.append(sentence)
        data_file.close()

        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        numericalized_sentence = [self.vocab.stoi["<SOS>"]]
        numericalized_sentence += self.vocab.numericalize(sentence)
        numericalized_sentence.append(self.vocab.stoi["<EOS>"])
        return torch.tensor(numericalized_sentence), torch.tensor(label)


if __name__ == "__main__":
    # vocab = Vocabulary("/Users/gaoyibo/Datasets/ml2020spring-hw4/")
    # vocab.build_vocabulary()
    # word_embedding_weight = vocab.get_wv()
    # torch.save(word_embedding_weight, './word_embedding_weight.pth')
    print("---------------------------")
