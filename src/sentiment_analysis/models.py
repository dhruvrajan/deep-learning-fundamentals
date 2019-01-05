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
    def __init__(self, word_vectors, inp, hid, out, lstm_dropout, bidirectional, freeze=False):
        super(SimpleLSTM, self).__init__()
        self.embedding_layer = PretrainedEmbeddingLayer(word_vectors, freeze=freeze)
        self.lstm = nn.LSTM(inp, hid, dropout=lstm_dropout, bidirectional=bidirectional)

        self.hidden_to_tag = nn.Linear(hid, out)
        self.hidden_state = self.init_hidden(hid)

    def init_hidden(self, hidden_size):
        return (torch.zeros(2, 1, hidden_size),
                torch.zeros(2, 1, hidden_size))

    def forward(self, sentence):
        embeddings = self.embedding_layer(sentence)
        # (seq_len, batch_size, inp_size)
        # (

        pack_padded_sequence
        lstm_out, self.hidden = self.lstm(embeddings.view(len(sentence), 1, -1), self.hidden)
        return F.log_softmax(self.hidden_to_tag(lstm_out.view(len(sentence), -1)), dim=1)
