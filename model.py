import torch.nn as nn


class RNN_Model(nn.Module):
    def __init__(self, args, word_embedding_weight):
        super(RNN_Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_embedding_weight, freeze=False)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim, args.num_layers, bidirectional=True, batch_first=True, dropout=args.dropout)
        self.fc = nn.Linear(args.hidden_dim * 2, args.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        out = self.sigmoid(out)
        return out