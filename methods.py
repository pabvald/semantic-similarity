import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


def avg_cosine(sentences1, sentences2, model):
    """ 
    Computes the cosine similarity between pairs of sentence
    embeddings, which are obtained computing the average of the 
    corresponding sentence's words embeddings found in the given
    model 

    :param sentences1: first set of sentences
    :param sentences2: second set of sentences 
    :param model: gensim model

    :returns: average cosine similarity of every pair of sentences 
    """
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

def remove_first_principal_component(X):
    """ Removes the First Principal Component of X """    
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX

def sif_cosine(sentences1, sentences2, model, frequencies={}, a=0.001):
    """
    Computes the cosine similarity between pairs of sentence
    embeddings, which are obtained computing a SIF average of the 
    corresponding sentence's words embeddings found in the given
    model 

    :param sentences1: first set of sentences
    :param sentences2: second set of sentences 
    :param model: gensim model
    :param frequencies: word frequencies
    :param a: 

    :returns: SIF-average cosine similarity of every pair of sentences 
    """
    vec_dim = model['the'].shape[0]
    total_freq = sum(frequencies.values())
    embeddings = []
    sims = []

    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = list(filter(lambda token: token in model, sent1))
        tokens2 = list(filter(lambda  token: token in model, sent2))
        
        if len(tokens1) == 0 or len(tokens2) == 0:           
            embedding1 = np.random.rand(vec_dim)  
            embedding2 = np.random.rand(vec_dim)
            
        else:
            weights1 = list(map(lambda token: a/ (a + frequencies.get(token,0)/total_freq), tokens1))
            weights2 = list(map(lambda token: a/ (a + frequencies.get(token,0)/total_freq), tokens2))       

            embedding1 = np.average(list(map(lambda token: model[token], tokens1)), axis=0, weights=weights1)
            embedding2 = np.average(list(map(lambda token: model[token], tokens2)), axis=0, weights=weights2)     
        
        embeddings.append(embedding1)
        embeddings.append(embedding2)
    
    embeddings = remove_first_principal_component(np.array(embeddings))
    
    for i in range(int(len(embeddings)/2)):
        sims.append(cosine_similarity(embeddings[i*2].reshape(1, -1), embeddings[i*2+1].reshape(1, -1))[0][0])

    return sims

def wmd(sentences1, sentences2, model):
    """ 
    Computes the Word Mover's distance between pairs of sentences. The 
    WMD is computed from the given model.
    :param sentences1: first set of sentences
    :param sentences2: second set of sentences 
    :param model: gensim model

    :returns: Word Mover's distance of every pair of sentences 
    """ 
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = list(filter(lambda token: token in model, sent1))
        tokens2 = list(filter(lambda token: token in model, sent2)) 
        d = -model.wmdistance(tokens1, tokens2)
        sims.append(max(d,-100))

    return sims
