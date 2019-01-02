from utils.indexer import Indexer
import numpy as np
import sys
import time
GLOVE_EMBEDDINGS = "data/glove_embeddings/glove.6B.300d-relativized.txt"

# Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
# that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
# word embedding files.
def read_word_embeddings(embeddings_file):
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    count = 0
    for line in f:
        if line.strip() != "":
            if count % 5000 == 0:
                print("loading glove embeddings... {}/17614\r".format(count))
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


if __name__ == '__main__':
    indexer, vectors = read_word_embeddings(GLOVE_EMBEDDINGS)