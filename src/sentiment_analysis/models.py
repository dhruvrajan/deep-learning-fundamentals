import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return F.log_softmax(self.fc4(F.sigmoid(self.fc3(F.relu(self.fc2(F.sigmoid(self.fc1(self.input_dropout(average_embeddings)))))))), dim=1)