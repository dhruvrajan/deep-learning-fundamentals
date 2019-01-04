from utils.indexer import Indexer
import numpy as np
import sys
import time

GLOVE_300 = "data/glove_embeddings/glove.6B.300d.txt"
GLOVE_300_RELATIVIZED = "data/glove_embeddings/glove.6B.300d-relativized.txt"
GLOVE_200 = "data/glove_embeddings/glove.6B.200d.txt"
GLOVE_200_RELATIVIZED = "data/glove_embeddings/glove.6B.200d-relativized.txt"
GLOVE_100 = "data/glove_embeddings/glove.6B.100d.txt"
GLOVE_100_RELATIVIZED = "data/glove_embeddings/glove.6B.100d-relativized.txt"
GLOVE_50 = "data/glove_embeddings/glove.6B.50d.txt"
GLOVE_50_RELATIVIZED = "data/glove_embeddings/glove.6B.50d-relativized.txt"


# Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
# that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
# word embedding files.
def read_word_embeddings(embeddings_file):
    f = open(embeddings_file)
    word_indexer = Indexer.create_indexer(with_symbols=True)
    vectors = []
    count = 0
    for line in f:
        if line.strip() != "":
            if count % 5000 == 0:
                print("loading glove embeddings... {}\r".format(count))
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx + 1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            # print repr(float_numbers)
            vector = np.array(float_numbers)
            word_indexer.add(word)
            vectors.append(vector)
            count += 1
            # print repr(word) + " : " + repr(vector)

    f.close()
    # Turn vectors into a 2-D numpy array
    np_vectors = np.array(vectors)
    np_vectors = np.vstack((np.zeros((2, np_vectors.shape[1])), np_vectors))

    print("Loaded Glove Embeddings", np_vectors.shape)
    return np_vectors, word_indexer


def relativize(vectors, glove_indexer: Indexer, word_indexer: Indexer, outfile=GLOVE_50_RELATIVIZED):
    new_vectors = []
    new_indexer = Indexer()
    for index in word_indexer.int2obj.keys():
        word = word_indexer.get_obj(index)
        if glove_indexer.has_obj(word):
            new_indexer.add(word)
            new_vectors.append(vectors[glove_indexer.get_idx(word)])

    out_vectors = np.vstack(new_vectors)

    with open(outfile, "w") as f:
        for i in range(len(new_indexer)):
            word = new_indexer.get_obj(i)
            f.write(word + " " + " ".join(map(str, out_vectors[i].tolist())) + "\n")

    print("relativized embeddings: {}, {} -> {}".format(len(glove_indexer), len(word_indexer), len(new_indexer)))

    return out_vectors, new_indexer


def case(item, dict):
    assert item in dict
    return dict[item]


def load_glove_embeddings(vector_size=300):
    source = case(vector_size, {
        300: GLOVE_300_RELATIVIZED,
        200: GLOVE_200_RELATIVIZED,
        100: GLOVE_100_RELATIVIZED,
        50: GLOVE_50_RELATIVIZED
    })

    return read_word_embeddings(source)


if __name__ == '__main__':
    indexer, vectors = load_glove_embeddings()
