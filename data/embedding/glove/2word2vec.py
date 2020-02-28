from gensim.scripts.glove2word2vec import glove2word2vec


if __name__ == '__main__':
    _ = glove2word2vec('./glove.840B.300d.txt', './glove.840B.300d.w2v.txt')