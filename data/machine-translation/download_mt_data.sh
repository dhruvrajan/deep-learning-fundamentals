#!/usr/bin/env bash

if [ -e "fra-eng.zip" ]; then
    echo "french-english translation data already exists; remove fra-eng.zip to download again."
else
    echo "downloading french-english machine-translation data from http://www.manythings.org/anki/fra-eng.zip"
    wget http://www.manythings.org/anki/fra-eng.zip
    unzip fra-eng.zip
    echo "finished downloading french-english translation data."
fi