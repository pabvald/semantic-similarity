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
PATH_TO_VEC = 'word_embeddings/glove/glove.6B.300d.txt'

# Pre-trained model characteristics 
VECTOR_DIMENSION = 300 
WORD2VEC_FORMAT = False

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

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
