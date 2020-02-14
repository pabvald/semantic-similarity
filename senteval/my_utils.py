import numpy as np 
import spacy

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