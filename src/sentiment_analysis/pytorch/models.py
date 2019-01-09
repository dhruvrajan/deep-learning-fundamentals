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
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
    def __init__(self, word_vectors, inp, hid, out, lstm_dropout, bidirectional, args):
        embedding_dim = args.embedding_size
        hidden_dim = hid
        tagset_size = out


        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = PretrainedEmbeddingLayer(word_vectors, freeze=args.freeze)#nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence, *args):
        # sentence = sentence[-1]
        embeds = self.word_embeddings(sentence)[-1]
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(embeds), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out[-1])#.view(len(embeds), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# class SimpleLSTM(nn.Module):
#     def __init__(self, word_vectors, inp, hid, out, lstm_dropout, bidirectional, args):
#         super(SimpleLSTM, self).__init__()
#         self.batch_size = args.batch_size
#         self.bidirectional = bidirectional
#         self.embedding_size = inp
#         self.hidden_size = hid
#         self.out = out
#         ##
#         self.embedding_layer = PretrainedEmbeddingLayer(word_vectors, freeze=args.freeze)
#         self.lstm = nn.LSTM(inp, hid, dropout=lstm_dropout, bidirectional=self.bidirectional)
#
#         self.hidden_to_tag = nn.Linear(hid * (2 if self.bidirectional else 1), out)
#         self.hidden_state = self.init_hidden(hid)
#
#         self.init_weights()
#
#         self.summary = {
#             "model_type": "SimpleLSTM",
#             "bidirectional": self.bidirectional,
#             "inp": self.embedding_size,
#             "hid": self.hidden_size,
#             "out": self.out,
#             "freeze_embeddings": args.freeze,
#             "lstm_dropout": lstm_dropout
#         }
#     def init_weights(self):
#         nn.init.xavier_uniform_(self.hidden_to_tag.weight)
#
#     def init_hidden(self, hidden_size, batch_size=None):
#         if not batch_size:
#             batch_size = self.batch_size
#         return (torch.zeros(2 if self.bidirectional else 1, batch_size, hidden_size),
#                 torch.zeros(2 if self.bidirectional else 1, batch_size, hidden_size))
#
#     def forward(self, sentence, input_lengths=None):
#         embeddings = self.embedding_layer(sentence).view(sentence.shape[1], sentence.shape[0], -1)
#         lstm_out, self.hidden_state = self.lstm(embeddings, self.hidden_state)
#         tag = self.hidden_to_tag(lstm_out[-1])
#         return F.log_softmax(tag, dim=1)
