#!/usr/bin/env bash

if [ -e "glove.6B.zip" ]; then
    echo "glove embeddings already exist"
else
    echo "downloading glove embeddings..."
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    echo "finished downloading glove embeddings."
fi