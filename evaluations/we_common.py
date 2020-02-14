# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
    Common methods use in the evaluation of different word embedding methods 
    whose pretrained vocabulary vectors are stored in text files where each
    line containes a word and its corresponding vector 
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging


def create_dictionary(sentences, threshold=0):
    """ Creates a dictionary with words that are present in the sentences """
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


def get_wordvec(path_to_vec, word2id, word2vec_format=True):
    """ Gets the word vectors from a vocabulary (glove, word2vec, fasttext...) """
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        if word2vec_format:
            next(f) #skip first line containing the vectors dimensions
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


def batcher_avg(params, batch):
    """ SentEval batchar method. The sentences embeddings are computed
        obtaining the average of its word embeddings """
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings
