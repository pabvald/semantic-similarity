import numpy as np 
import spacy
from scipy.stats import spearmanr, pearsonr

def preprocess(sentences, lowercase=True, stop_words=True, punctuation=True,
                                                 only_ascii=True, lemmatization=True):
    """ Preprocesses the given sentences applying the specified filters 
        and extracting the tokens that verify those filters """

    nlp = spacy.load("en_core_web_sm")
    preprocessed_sentences = []  
    
    for doc in nlp.pipe(sentences, disable=["tagger", "parser", "ner"]):
        tokens = doc.doc
        
        if stop_words: 
            tokens =  list(filter(lambda t : not t.is_stop, tokens))
        
        if punctuation:
            tokens = list(filter(lambda t: not t.is_punct, tokens))
        
        if only_ascii:
            tokens = list(filter(lambda t: t.is_ascii, tokens))
    
        if lemmatization: 
            tokens =  list(map(lambda t: t.lemma_, tokens))
        else:
            tokens = list(map(lambda t: t.text, tokens))
    
        if lowercase:
            tokens = list(map(lambda t: t.lower(), tokens))
        
        preprocessed_sentences.append(np.array(tokens))

    return np.array(preprocessed_sentences)

def evaluate(task, methods):
    """ Computes the weigthed Pearson and Spearman correlations of a STS task 
        using the given methods"""
    pearson_correlations = {}
    spearman_correlations = {}
    
    for label, method in methods:
        task_pearson = []
        task_spearman = []
        task_weights = [] 
        for dataset in task.keys():
            sentences1, sentences2, gs = task[dataset]
            task_weights.append(len(gs))
            sims = method(sentences1, sentences2)
            task_pearson.append(pearsonr(sims, gs)[0])
            task_spearman.append(spearmanr(sims, gs)[0])

        wpearson = sum(task_pearson[i] * task_weights[i] / sum(task_weights) for i in range(len(task_weights)))
        wspearman =  sum(task_spearman[i] * task_weights[i] / sum(task_weights) for i in range(len(task_weights)))
       
        pearson_correlations[label] = wpearson
        spearman_correlations[label] = wspearman
        
    return pearson_correlations, spearman_correlations