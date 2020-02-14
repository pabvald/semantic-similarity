# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
QUORA: Quora Question Pairs duplicates detection 
'''
from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import pandas as pd 
import io

from senteval.tools.validation import KFoldClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class QuoraEval(object):
    
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : Quora *****\n\n')
        self.seed = seed

        train, test = self.loadFile(os.path.join(task_path, 'quora_duplicate_questions.csv'))
        self.quora_data = {'train': train, 'test': test}


    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.quora_data['train']['X_Q1'] + \
                  self.quora_data['train']['X_Q2'] + \
                  self.quora_data['test']['X_Q1'] + self.quora_data['test']['X_Q2']
        return prepare(params, samples)

    def loadFile(self, fpath):       
        # Read csv file
        csv_file = pd.read_csv(fpath, sep=',')
        # Drop invalid rows
        clean_csv_file = csv_file.dropna(how='any', axis=0, subset=['question1', 'question2', 'is_duplicate'])

        q1 = clean_csv_file.values[:, 3]
        q2 = clean_csv_file.values[:, 4]
        y = np.array(clean_csv_file.values[:, 5], dtype='int64')
        
        # Holdout 66%
        q1_train, q1_test, q2_train, q2_test, y_train, y_test = train_test_split(q1, q2, y, test_size=0.33, stratify=y, shuffle=True)
        
        quora_train_data = {'X_Q1': q1_train.tolist(), 'X_Q2': q2_train.tolist(), 'y': y_train.tolist()}
        quora_test_data = {'X_Q1': q1_test.tolist(), 'X_Q2': q2_test.tolist(), 'y': y_test.tolist()}

        return quora_train_data, quora_test_data

    def run(self, params, batcher):
        quora_embed = {'train': {}, 'test': {}}

        for key in self.quora_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.quora_data[key]['X_Q1'],
                                       self.quora_data[key]['X_Q2'],
                                       self.quora_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            text_data['Q1'] = [x for (x, y, z) in sorted_corpus]
            text_data['Q2'] = [y for (x, y, z) in sorted_corpus]
            text_data['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['Q1', 'Q2']:
                quora_embed[key][txt_type] = []
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    quora_embed[key][txt_type].append(embeddings)
                quora_embed[key][txt_type] = np.vstack(quora_embed[key][txt_type])
            quora_embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = quora_embed['train']['Q1']
        trainB = quora_embed['train']['Q2']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = quora_embed['train']['y']

        # Test
        testA = quora_embed['test']['Q1']
        testB = quora_embed['test']['Q2']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = quora_embed['test']['y']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = KFoldClassifier(train={'X': trainF, 'y': trainY},
                              test={'X': testF, 'y': testY}, config=config)

        devacc, testacc, yhat = clf.run()
        testf1 = round(100*f1_score(testY, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for Quora.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainA), 'ntest': len(testA)}
