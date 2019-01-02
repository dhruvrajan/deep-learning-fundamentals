from nltk import word_tokenize, WordNetLemmatizer
import re

TRAIN_PATH = "pytorch/sentiment_analysis/data/train.txt"
DEV_PATH = "pytorch/sentiment_analysis/data/dev.txt"

class Indexer:
    UNK_SYMBOL = "<UNK>"
    PAD_SYMBOL = "<PAD>"

    def __init__(self, obj_type=str, idx_type=int):
        self.obj2int = {}
        self.int2obj = {}

        assert obj_type != idx_type
        self.obj_type = obj_type
        self.idx_type = idx_type

    def has_idx(self, idx):
        assert type(idx) == self.idx_type
        return idx in self.int2obj

    def has_obj(self, obj):
        assert type(obj) == self.obj_type
        return obj in self.obj2int

    def get_obj(self, idx):
        if self.has_idx(idx):
            return self.int2obj[idx]
        return None

    def get_idx(self, obj):
        if self.has_obj(obj):
            return self.obj2int[obj]
        elif self.has_obj(Indexer.UNK_SYMBOL):
            return self.obj2int[Indexer.UNK_SYMBOL]
        return -1

    def add(self, obj):
        if self.has_obj(obj):
            return self.get_idx(obj)

        idx = len(self.obj2int)
        self.obj2int[obj] = idx
        self.int2obj[idx] = obj
        return idx

    @staticmethod
    def create_indexer(with_symbols=True):
        indexer = Indexer()
        if with_symbols:
            indexer.add(Indexer.PAD_SYMBOL)
            indexer.add(Indexer.UNK_SYMBOL)

        return indexer

    def __getitem__(self, item):
        assert type(item) in (self.obj_type, self.idx_type)

        if type(item) == self.obj_type:
            return self.get_idx(item)

        elif type(item) == self.idx_type:
            return self.get_obj(item)


class SentimentExample:
    def __init__(self, sentiment, indexed_words, words=[]):
        self.sentiment = sentiment
        self.indexed_words = indexed_words
        self.words = words

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

            sentiments.append(sentiment)
            sentences.append(word_tokenize(sentence))

            for word in sentences[-1]:
                word_indexer.add(word.strip().lower())

    for sentence in sentences:
        indexed_words = [word_indexer.get_idx(word.strip().lower()) for word in sentence]
        indexed_sentences.append(indexed_words)


    return [SentimentExample(*pair) for pair in zip(sentiments, indexed_sentences, sentences)], word_indexer


class SentimentDataset:
    def __init__(self, examples=None, word_indexer=None):
        self.examples = examples
        self.word_indexer = word_indexer

    @staticmethod
    def load_from(path):
        return SentimentDataset(*read_sentiment_data(path))


if __name__ == '__main__':
    # Run Indexer Test
    word_indexer = Indexer.create_indexer()
    word_indexer.add("fluffy")
    word_indexer.add("bunny")

    print(word_indexer["fluffy"], word_indexer[3])
    print(word_indexer["bunny"], word_indexer[1])
    print(word_indexer["rabbit"], word_indexer[2])

    train_data = SentimentDataset.load_from(TRAIN_PATH)
    dev_data = SentimentDataset.load_from(DEV_PATH)

    print("finished loading data.")