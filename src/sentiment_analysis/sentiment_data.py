import re
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from nltk import word_tokenize, WordNetLemmatizer
from torch.utils.data import TensorDataset

from utils.glove_embeddings import load_glove_embeddings
from utils.indexer import Indexer

TRAIN_PATH = "data/sentiment-analysis/train.txt"
DEV_PATH = "data/sentiment-analysis/dev.txt"


class SentimentExample:
    def __init__(self, sentiment, indexed_words, words):
        self.sentiment = sentiment
        self.indexed_words = indexed_words
        self.words = words

    def __repr__(self):
        return str(self.sentiment) + ", " + str(self.words[:min(len(self.words), 50)]) + "...\n"


def clean_str(string):
    wnl = WordNetLemmatizer()
    string = " ".join(list(map(wnl.lemmatize, word_tokenize(string))))

    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`\-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\-", " - ", string)
    # We may have introduced double spaces, so collapse these down
    string = re.sub(r"\s{2,}", " ", string)
    return string


def read_sentiment_data(path, relativized_indexer=None):
    sentences = []
    indexed_sentences = []
    sentiments = []
    word_indexer = Indexer.create_indexer() if not relativized_indexer else relativized_indexer
    with open(path, encoding="iso8859") as f:
        for line in f:
            sentiment, sentence = line.strip().split("\t")
            sentence = " ".join(list(filter(lambda x: x != '', clean_str(sentence).rstrip().split(" "))))

            sentiments.append(int(sentiment))
            sentences.append(word_tokenize(sentence))

            if not relativized_indexer:
                for word in sentences[-1]:
                    word_indexer.add(word.strip().lower())

    max_len = 0
    for sentence in sentences:
        indexed_words = [word_indexer.get_idx(word.strip().lower()) for word in sentence]
        indexed_sentences.append(indexed_words)
        if len(indexed_words) > max_len:
            max_len = len(indexed_words)

    return [SentimentExample(*pair) for pair in zip(sentiments, indexed_sentences, sentences)], word_indexer


class SentimentDataset:
    def __init__(self, examples=None, word_indexer=None):
        self.examples = examples
        self.word_indexer = word_indexer

    def create_tensor_dataset(self, batch_size):
        indexed_sentences = [ex.indexed_words for ex in self.examples]
        input_lengths = [len(sentence) for sentence in indexed_sentences]
        sentiments = [ex.sentiment for ex in self.examples]

        sentence_tensors = [torch.Tensor(sentence).long() for sentence in indexed_sentences]
        padded_sentences = pad_sequence(sentence_tensors, batch_first=True)

        until = (padded_sentences.shape[0] % batch_size) * -1

        X = padded_sentences[:until]
        y = torch.Tensor(sentiments).long()[:until]
        input_lengths_tensor = torch.Tensor(input_lengths).long()[:until]


        return TensorDataset(X, y, input_lengths_tensor)

    @staticmethod
    def load_from(path, word_indexer=None):
        examples, _ = read_sentiment_data(path, word_indexer)
        return SentimentDataset(examples, word_indexer)


class WordVectors:
    def __init__(self, vectors, indexer):
        self.word_indexer = indexer
        self.vectors = vectors

    def get_embedding(self, word):
        if self.word_indexer.has_obj(word):
            return self.vectors[self.word_indexer[word]]

        return self.vectors[self.word_indexer[Indexer.UNK_SYMBOL]]


def load_sentiment_data(embedding_size):
    word_vectors = WordVectors(*load_glove_embeddings(embedding_size))

    train_dataset = SentimentDataset.load_from(TRAIN_PATH, word_vectors.word_indexer)
    dev_dataset = SentimentDataset.load_from(DEV_PATH, word_vectors.word_indexer)

    return train_dataset, dev_dataset, word_vectors

if __name__ == '__main__':
    load_sentiment_data(50)