from nltk import word_tokenize, WordNetLemmatizer
import re
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch import Tensor
import torch
from utils.indexer import Indexer

TRAIN_PATH = "data/sentiment-analysis/train.txt"
DEV_PATH = "data/sentiment-analysis/dev.txt"


class SentimentExample:
    def __init__(self, sentiment, indexed_words, words, pad_to=None):
        self.sentiment = sentiment
        self.indexed_words = indexed_words
        self.words = words
        self.pad_to = pad_to if pad_to else len(indexed_words)


    def __repr__(self):
        return str(self.sentiment) + ", " + str(self.words[:min(len(self.words), 50)]) + "...\n"


def apply_index(sentence, word_index, padding=None):
    tokens = word_tokenize(sentence)


def create_example(line, indexer):
    sentiment, sentence = line.strip().split("\t")
    tokenized = word_tokenize(sentence)


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


def read_sentiment_data(path):
    sentences = []
    indexed_sentences = []
    sentiments = []
    word_indexer = Indexer.create_indexer()
    with open(path, encoding="iso8859") as f:
        for line in f:
            sentiment, sentence = line.strip().split("\t")
            sentence = " ".join(list(filter(lambda x: x != '', clean_str(sentence).rstrip().split(" "))))

            sentiments.append(int(sentiment))
            sentences.append(word_tokenize(sentence))

            for word in sentences[-1]:
                word_indexer.add(word.strip().lower())

    max_len = 0
    for sentence in sentences:
        indexed_words = [word_indexer.get_idx(word.strip().lower()) for word in sentence]
        indexed_sentences.append(indexed_words)
        if len(indexed_words) > max_len:
            max_len = len(indexed_words)

    return [SentimentExample(*pair) for pair in zip(sentiments, indexed_sentences, sentences)], word_indexer, max_len


class SentimentDataset:
    def __init__(self, examples=None, word_indexer=None):
        self.examples = examples
        self.word_indexer = word_indexer
        self.max_len = len(max(self.examples, key=lambda ex: len(ex.indexed_words)).indexed_words)

        # def to_tensor_dataset(self):
        #     indexed_words = [ex.indexed_words for ex in self.examples]
        # sentiments = [ex.sentiment for ex in self.examples]

        # return TensorDataset(torch)

    def make_sentence_tensor(self, indexed_words, pad_to):
        assert len(indexed_words) <= pad_to
        tensor = torch.Tensor(indexed_words).long()
        padding = pad_to - tensor.shape[0]
        pad_value = self.word_indexer.get_idx(Indexer.PAD_SYMBOL)
        return F.pad(tensor, [0, padding], mode="constant", value=pad_value)

    def create_tensor_dataset(self):
        indexed_sentences = [ex.indexed_words for ex in self.examples]
        sentiments = [ex.sentiment for ex in self.examples]

        sentence_tensors = [self.make_sentence_tensor(sentence, self.max_len) for sentence in indexed_sentences]

        X = torch.stack(sentence_tensors)
        y = torch.Tensor(sentiments).long()

        return TensorDataset(X, y)


    @staticmethod
    def load_from(path):
        return SentimentDataset(*read_sentiment_data(path)[:-1])


class WordVectors:
    def __init__(self, vectors, indexer):
        self.word_indexer = indexer
        self.vectors = vectors

    def get_embedding(self, word):
        if self.word_indexer.has_obj(word):
            return self.vectors[self.word_indexer[word]]

        return self.vectors[self.word_indexer[Indexer.UNK_SYMBOL]]




def load_sentiment_data():
    train_dataset = SentimentDataset.load_from(TRAIN_PATH)
    dev_dataset = SentimentDataset.load_from(DEV_PATH)
    return train_dataset, dev_dataset

if __name__ == '__main__':
    train_data = SentimentDataset.load_from(TRAIN_PATH)
    train_data.create_tensor_dataset()
    dev_data = SentimentDataset.load_from(DEV_PATH)

    print("finished loading data.")
