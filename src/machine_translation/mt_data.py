import re
from nltk import WordNetLemmatizer, word_tokenize

from utils.indexer import Indexer
import logging

MT_DATA_PATH = "data/machine-translation/fra.txt"
logger = logging.Logger("MT Data")


class MTExample:
    def __init__(self, source, target, source_lang="eng", target_lang="fr"):
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.source_sentence = source
        self.target_sentence = target

    def raw(self):
        return self.source_sentence, self.target_sentence

    def clean(self):
        return MTExample.cleaned(self.source_sentence), MTExample.cleaned(self.target_sentence)

    def words(self):
        source, target = self.clean()
        return word_tokenize(source), word_tokenize(target)

    @staticmethod
    def cleaned(sentence):
        return " ".join(list(filter(lambda x: x != '', clean_str(sentence).rstrip().split(" "))))

    def __repr__(self):
        return "( \"" + self.source_sentence + "\", \"" + self.target_sentence + "\" )"


class MTDataset:
    def __init__(self, mt_examples, source_indexer, target_indexer):
        self.mt_examples = mt_examples
        self.source_indexer = source_indexer
        self.target_indexer = target_indexer

    @staticmethod
    def from_file(path=MT_DATA_PATH):
        return MTDataset(*read_and_index_mt_data(path))

    def __len__(self):
        return len(self.mt_examples)


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
    string = re.sub(r"\s{2,}", " ", string).lower()
    return string


def read_and_index_mt_data(path=MT_DATA_PATH):
    logger.info("Reading and Indexing MT data")
    examples = []
    source_indexer = Indexer.create_indexer()
    target_indexer = Indexer.create_indexer()
    with open(path) as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print("loading mt examples (%i)" % i)

            source, target = line.strip().split("\t")
            mt_example = MTExample(source, target)
            examples.append(mt_example)

            source_words, target_words = mt_example.words()
            source_indexer.add_all(source_words)
            target_indexer.add_all(target_words)

    return examples, source_indexer, target_indexer


def load_mt_data():
    return MTDataset.from_file()


if __name__ == '__main__':
    dataset = MTDataset.from_file()
    print("Loaded MT dataset; %i examples." % len(dataset))
