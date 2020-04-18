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
import os
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


def load_file_STS(path, datasets, preprocessing, verbose=False):
    """ Loads a STS test file and preprocesses its sentences """
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


def load_SICK(path, preprocessing, verbose=False):
    """ Loads the SICK train, dev and test files and preprocess its sentences """
    if verbose:
        print('\n\n***** Task: SICK-Relatedness*****\n')
    sick_all = {}
    sick_train = load_file_SICK(os.path.join(path, 'SICK_train.txt'), preprocessing)
    sick_dev = load_file_SICK(os.path.join(path, 'SICK_trial.txt'), preprocessing)
    sick_test = load_file_SICK(os.path.join(path, 'SICK_test_annotated.txt'), preprocessing)
    
    sick_all['train'] = sick_train
    sick_all['test'] = sick_test
    sick_all['dev'] = sick_dev
    
    return sick_all


def load_file_SICK(path, preprocessing):
    """ Loads a SICK file and preprocess its sentences """
    skip_first_line = True
    sent1 = []
    sent2 = []
    sim = []
    # Read file
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if skip_first_line:
                skip_first_line = False
            else:
                text = line.strip().split('\t')
                sent1.append(text[1])
                sent2.append(text[2])
                sim.append(text[3])

    sent1 = preprocess(sent1, **preprocessing)
    sent2 = preprocess(sent2, **preprocessing)
    sim = [float(s) for s in sim]
    
    return (sent1, sent2, sim)


def load_sts_12(path, preprocessing, verbose=False):
    """ Loads the SemEval-2012's Semantic Textual Similarity task"""
    if verbose:
        print('\n\n***** TASK: STS12 *****\n')
    datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                        'surprise.OnWN', 'surprise.SMTnews']
    return load_file_STS('{}/STS12-en-test'.format(path), datasets, preprocessing, verbose=verbose)


def load_sts_13(path, preprocessing, verbose=False):
    """ Loads the SemEval-2013's Semantic Textual Similarity task"""
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    if verbose:
        print('\n\n***** TASK: STS13 (-SMT) ***\n')
    datasets = ['FNWN', 'headlines', 'OnWN']
    return load_file_STS('{}/STS13-en-test'.format(path), datasets, preprocessing, verbose=verbose)


def load_sts_14(path, preprocessing, verbose=False):
    """ Loads the SemEval-2014's Semantic Textual Similarity task"""
    if verbose:
        print('\n\n***** TASK: STS14 *****\n')
    datasets = ['deft-forum', 'deft-news', 'headlines',
                        'images', 'OnWN', 'tweet-news']
    return load_file_STS('{}/STS14-en-test'.format(path), datasets, preprocessing, verbose=verbose)


def load_sts_15(path, preprocessing, verbose=False):
    """ Loads the SemEval-2015's Semantic Textual Similarity task"""
    if verbose:
        print('\n\n***** TASK: STS15 *****\n')
    datasets = ['answers-forums', 'answers-students',
                        'belief', 'headlines', 'images']
    return load_file_STS('{}/STS15-en-test'.format(path), datasets, preprocessing, verbose=verbose)


def load_sts_16(path, preprocessing, verbose=False):
    """ Loads the SemEval-2016's Semantic Textual Similarity task"""
    if verbose:
        print('\n\n***** TASK: STS16 *****\n')
    datasets = ['answer-answer', 'headlines', 'plagiarism',
                        'postediting', 'question-question']
    return load_file_STS('{}/STS16-en-test'.format(path), datasets, preprocessing, verbose=verbose)

