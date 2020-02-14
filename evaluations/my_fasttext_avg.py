# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
from we_common import get_wordvec, create_dictionary, batcher_avg

# PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
#PATH_TO_VEC = 'word_embeddings/fasttext/wiki-news-300d-1M-subword.vec'
PATH_TO_VEC = 'word_embeddings/fasttext/crawl-300d-2M.vec'

# Pre-trained model characteristics 
VECTOR_DIMENSION = 300 
WORD2VEC_FORMAT = True

# Preprocessing parameters 
LOWERCASE = True # Lowercase the text
LEMMATIZATION = True # Substitute words by their lemma
PUNCTUATION = True # Remove punctuation characters
STOP_WORDS = True # Remove stop words
ONLY_ASCII = True # Remove non-ascii text

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
params_senteval['preprocessing'] = {'lowercase': LOWERCASE, 'lemmatization': LEMMATIZATION, 'stop_words': STOP_WORDS, 
                                     'punctuation': PUNCTUATION, 'only_ascii': ONLY_ASCII } 

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


# SentEval methods 
def prepare(params, samples):
    """ SentEval prepare method """
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id, word2vec_format=WORD2VEC_FORMAT)
    params.wvec_dim = VECTOR_DIMENSION
    return
    

def batcher(params, batch):
    """ SentEval batcher method """
    embeddings = batcher_avg(params, batch)
    return embeddings


if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12']
    results = se.eval(transfer_tasks)
    print(results)
