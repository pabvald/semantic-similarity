import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def avg_cosine(sentences1, sentences2, model):
    """ Computes the cosine similarity between pairs of sentence
        embeddings, which are obtained computing the average of the 
        corresponding sentence's words embeddings found in the given
        model """
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = list(filter(lambda token: token in model, sent1))
        tokens2 = list(filter(lambda token: token in model, sent2))      
        
        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)

        else:    
            embedding1 = np.average(list(map(lambda token: model[token], tokens1)), axis=0).reshape(1, -1)
            embedding2 = np.average(list(map(lambda token: model[token], tokens2)), axis=0).reshape(1, -1)

            sim = cosine_similarity(embedding1, embedding2)[0][0]
            sims.append(sim)

    return sims

def wmd(sentences1, sentences2, model):
    """ Computes the Word Mover's distance between pairs of sentences. The 
        WMD is computed from the given model """ 
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = list(filter(lambda token: token in model, sent1))
        tokens2 = list(filter(lambda token: token in model, sent2)) 
        d = -model.wmdistance(tokens1, tokens2)
        sims.append(max(d,-100))

    return sims