import torch.nn as nn
from sentiment_analysis.sentiment_utils import load_sentiment_data

class PretrainedWordEmbeddings(nn.Module):

    def __init__(self, word_vectors):
        super(PretrainedWordEmbeddings, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(wo)

class FFNN(nn.Module):

    def __init__(self):
        super(FFNN, self).__init__()




def train_ffnn():
    train_dataset, dev_dataset = load_sentiment_data()