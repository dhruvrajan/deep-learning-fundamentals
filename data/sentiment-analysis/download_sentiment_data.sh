#!/usr/bin/env bash

if [ -e "./data/mini2-distrib.tgz" ]; then
    echo "sentiment data already exists; remove mini2-distrib.tgz to download again."
else
    echo "downloading sentiment analysis data from Prof. Greg Durrett's 388..."
    wget https://www.cs.utexas.edu/~gdurrett/courses/fa2018/mini2-distrib.tgz
    tar -xf mini2-distrib.tgz
    mv mini2-distrib/data/train.txt .
    mv mini2-distrib/data/dev.txt .
    rm -r mini2-distrib
    echo "finished downloading sentiment analysis data."
fi