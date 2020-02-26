# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# SENTEVAL_LICENSE file in the root directory of this source tree.
#


"""
    This module contains different functions to load the test datasets
    of the SemEval's STS tasks from 2012 to 2016. This functions are based
    on those found in the SentEval toolkit.
"""


import io
import csv
import numpy as np
from utils import preprocess


def load_frequencies(path):
    """ Loads the word frequencies """
    frequencies = {}
    with open(path) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t")
        for row in tsv_reader: 
            frequencies[row[0]] = int(row[1])
        
    return frequencies

def load_file(path, datasets, preprocessing, verbose=False):
    """ Loads and STS file and pre-processes its sentences """
    data = {}

    for dataset in datasets:
        # Load sentences pairs
        sent1, sent2 = zip(*[l.split("\t") for l in
                            io.open(path + '/STS.input.%s.txt' % dataset,
                                    encoding='utf8').read().splitlines()])
        # Load Gold Standard files (similarity scores)
        raw_scores = np.array([x for x in
                                io.open(path + '/STS.gs.%s.txt' % dataset,
                                        encoding='utf8')
                                .read().splitlines()])
        # Consider only pairs with a score
        not_empty_idx = raw_scores != ''

        gs_scores = [float(x) for x in raw_scores[not_empty_idx]]

        # Preprocess sentences
        if verbose:
            print("Preprocessing -{}-".format(dataset))
        sent1 = preprocess(sent1, **preprocessing)[not_empty_idx]
        sent2 = preprocess(sent2, **preprocessing)[not_empty_idx]
        if verbose:
            print("-{}- preprocessed correctly".format(dataset))
        
        # Sort data by length to minimize padding in batcher
        sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                key=lambda z: (len(z[0]), len(z[1]), z[2]))
        sent1, sent2, gs_scores = map(list, zip(*sorted_data))

        data[dataset] = (sent1, sent2, gs_scores)
        
    return data

def load_sts_12(path, preprocessing, verbose=False):
    """ Loads the SemEval-2012's Semantic Textual Similarity task"""
    if verbose:
        print('\n***** TASK: STS12 *****\n')
    datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                        'surprise.OnWN', 'surprise.SMTnews']
    return load_file('{}/STS12-en-test'.format(path), datasets, preprocessing, verbose=verbose)

def load_sts_13(path, preprocessing, verbose=False):
    """ Loads the SemEval-2013's Semantic Textual Similarity task"""
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    if verbose:
        print('\n***** TASK: STS13 (-SMT) ***\n\n')
    datasets = ['FNWN', 'headlines', 'OnWN']
    return load_file('{}/STS13-en-test'.format(path), datasets, preprocessing, verbose=verbose)

def load_sts_14(path, preprocessing, verbose=False):
    """ Loads the SemEval-2014's Semantic Textual Similarity task"""
    if verbose:
        print('\n***** TASK: STS14 *****\n')
    datasets = ['deft-forum', 'deft-news', 'headlines',
                        'images', 'OnWN', 'tweet-news']
    return load_file('{}/STS14-en-test'.format(path), datasets, preprocessing, verbose=verbose)

def load_sts_15(path, preprocessing, verbose=False):
    """ Loads the SemEval-2015's Semantic Textual Similarity task"""
    if verbose:
        print('\n***** TASK: STS15 *****\n')
    datasets = ['answers-forums', 'answers-students',
                        'belief', 'headlines', 'images']
    return load_file('{}/STS15-en-test'.format(path), datasets, preprocessing, verbose=verbose)

def load_sts_16(path, preprocessing, verbose=False):
    """ Loads the SemEval-2016's Semantic Textual Similarity task"""
    if verbose:
        print('\n***** TASK: STS16 *****\n')
    datasets = ['answer-answer', 'headlines', 'plagiarism',
                        'postediting', 'question-question']
    return load_file('{}/STS16-en-test'.format(path), datasets, preprocessing, verbose=verbose)

