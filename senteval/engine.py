# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from senteval.mrpc import MRPCEval
from senteval.sick import SICKEntailmentEval, SICKRelatednessEval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval
from senteval.sst import SSTEval
from senteval.quora import QuoraEval
from senteval.mytrec import MyTrecEval
from senteval.probing import *



class SE(object):

    def __init__(self, params, batcher, preprocessing=None, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold
       

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['MRPC', 'Quora', 'MYTREC', 'SICKRelatedness', 'STSBenchmark', 'SICKEntailment', 
                           'STS12', 'STS13', 'STS14', 'STS15', 'STS16',  'Length',
                           'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                           'SubjNumber', 'ObjNumber', 'OddManOut','CoordinationInversion' ]
        
        self.preprocessing = {'lowercase':  True, 'stop_words': True, 'punctuation': True, 'only_ascii': True, 
                        'lemmatization': True} if not preprocessing else preprocessing


    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # # Duplicate/Paraphrase detection
        # if name == 'MYTREC':
        #     self.evaluation = MyTrecEval(tpath + '/downstream/MYTREC', self.preprocessing, seed=self.params.seed)
        # elif name == 'MRPC':
        #     self.evaluation = MRPCEval(tpath + '/downstream/MRPC', self.preprocessing, seed=self.params.seed)
        # elif name == 'Quora':
        #     self.evaluation = QuoraEval(tpath + '/downstream/QUORA', seed =self.params.seed)
        
        # # Semantic text similarity
        # elif name == 'SICKRelatedness':
        #     self.evaluation = SICKRelatednessEval(tpath + '/downstream/SICK', self.preprocessing, seed=self.params.seed)
        # elif name == 'STSBenchmark':
        #     self.evaluation = STSBenchmarkEval(tpath + '/downstream/STS/STSBenchmark', self.preprocessing, seed=self.params.seed)
        # elif name == 'SICKEntailment':
        #     self.evaluation = SICKEntailmentEval(tpath + '/downstream/SICK', self.preprocessing, seed=self.params.seed)
        if name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            fpath = name + '-en-test'
            self.evaluation = eval(name + 'Eval')(tpath + '/downstream/STS/' + fpath, self.preprocessing, seed=self.params.seed)

        # # Probing Tasks
        # elif name == 'Length':
        #         self.evaluation = LengthEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)
        # elif name == 'WordContent':
        #         self.evaluation = WordContentEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)
        # elif name == 'Depth':
        #         self.evaluation = DepthEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)
        # elif name == 'TopConstituents':
        #         self.evaluation = TopConstituentsEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)
        # elif name == 'BigramShift':
        #         self.evaluation = BigramShiftEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)
        # elif name == 'Tense':
        #         self.evaluation = TenseEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)
        # elif name == 'SubjNumber':
        #         self.evaluation = SubjNumberEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)
        # elif name == 'ObjNumber':
        #         self.evaluation = ObjNumberEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)
        # elif name == 'OddManOut':
        #         self.evaluation = OddManOutEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)
        # elif name == 'CoordinationInversion':
        #         self.evaluation = CoordinationInversionEval(tpath + '/probing', self.preprocessing, seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
