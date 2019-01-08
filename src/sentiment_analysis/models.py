import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class PretrainedEmbeddingLayer(nn.Module):
    def __init__(self, word_vectors, freeze=False):
        super(PretrainedEmbeddingLayer, self).__init__()
        embeddings_tensor = torch.Tensor(word_vectors.vectors).float()
        self.embedding = nn.Embedding.from_pretrained(embeddings_tensor, freeze=freeze)

    def forward(self, sentence):
        for idx in sentence:
            if not (idx < self.embedding.num_embeddings).sum() == sentence.shape[1]:
                print("we have praablem", idx.max().item(), self.embedding.num_embeddings)
        return self.embedding(sentence)


class SimpleFFNN(nn.Module):

    def __init__(self, word_vectors, inp, hid1, hid2, hid3, out, freeze=False):
        super(SimpleFFNN, self).__init__()
        self.word_vectors = word_vectors

        # Pre-trained GLOVE Embeddings
        self.embedding_layer = PretrainedEmbeddingLayer(word_vectors, freeze=freeze)
        self.input_dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(inp, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, hid3)
        self.fc4 = nn.Linear(hid3, out)
        self.init_weights()

        self.summary = {
            "model_type": "SimpleFFNN",
            "layers": [inp, hid1, hid2, hid3, out],
            "freeze_embeddings": freeze,
        }

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, sentence):
        average_embeddings = torch.mean(self.embedding_layer(sentence), dim=1)
        return F.log_softmax(
            self.fc4(F.tanh(self.fc3(F.sigmoid(self.fc2(F.tanh(self.fc1(self.input_dropout(average_embeddings)))))))),
            dim=1)


class SimpleLSTM(nn.Module):
    def __init__(self, word_vectors, inp, hid, out, lstm_dropout, bidirectional, args):
        super(SimpleLSTM, self).__init__()
        self.batch_size = args.batch_size
        self.bidirectional = bidirectional
        self.embedding_size = inp
        self.hidden_size = hid
        self.out = out
        self.embedding_layer = PretrainedEmbeddingLayer(word_vectors, freeze=args.freeze)
        self.lstm = nn.LSTM(inp, hid, dropout=lstm_dropout, bidirectional=self.bidirectional)

        self.hidden_to_tag = nn.Linear(hid * (2 if self.bidirectional else 1), out)
        self.hidden_state = self.init_hidden(hid)

        self.summary = {
            "model_type": "SimpleLSTM",
            "bidirectional": self.bidirectional,
            "inp": self.embedding_size,
            "hid": self.hidden_size,
            "out": self.out,
            "freeze_embeddings": args.freeze,
            "lstm_dropout": lstm_dropout
        }

    def init_hidden(self, hidden_size, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        return (torch.zeros(2 if self.bidirectional else 1, batch_size, hidden_size),
                torch.zeros(2 if self.bidirectional else 1, batch_size, hidden_size))

    def forward(self, sentence):
        embeddings = self.embedding_layer(sentence).view(sentence.shape[1], sentence.shape[0], -1)
        lstm_out, self.hidden_state = self.lstm(embeddings, self.hidden_state)
        tag = self.hidden_to_tag(self.hidden_state[0]).view(64, 2)
        return F.log_softmax(tag, dim=1)
