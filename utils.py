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


def evaluate(corpus, methods):
    """ Computes the weigthed Pearson and Spearman correlations of a STS corpus 
        using the given methods"""
    pearson_correlations = {}
    spearman_correlations = {}
    
    for label, method in methods:
        corpus_pearson = []
        corpus_spearman = []
        corpus_weights = [] 
        for dataset in corpus.keys():
            sentences1, sentences2, gs = corpus[dataset]
            corpus_weights.append(len(gs))
            sims = method(sentences1, sentences2)
            corpus_pearson.append(pearsonr(sims, gs)[0])
            corpus_spearman.append(spearmanr(sims, gs)[0])

        wpearson = sum(corpus_pearson[i] * corpus_weights[i] / sum(corpus_weights) for i in range(len(corpus_weights)))
        wspearman = sum(corpus_spearman[i] * corpus_weights[i] / sum(corpus_weights) for i in range(len(corpus_weights)))
       
        pearson_correlations[label] = wpearson
        spearman_correlations[label] = wspearman
        
    return pearson_correlations, spearman_correlations


def get_frequencies(corpus, threshold=0):
    """ Computes the frequencies of a corpus"""
    freqs = {}
    for dataset in corpus.keys():
        sentences1, sentences2, gs = corpus[dataset]
        
        for sent in (sentences1 + sentences2):
            for word in sent:
                freqs[word] = freqs.get(word, 0) + 1

        if threshold > 0:
            new_freqs = {}
            for word in freqs:
                if freqs[word] >= threshold:
                    new_freqs[word] = freqs[word]
            freqs = new_freqs
        freqs['<s>'] = 1e9 + 4
        freqs['</s>'] = 1e9 + 3
        freqs['<p>'] = 1e9 + 2
        
    return freqs