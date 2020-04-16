# Comparativa de vectores pre-entrenados de Word2Vec, GloVe y FastText para medir la similaridad semántica entre pares de oraciones


- **data/**
    - **datatsets/**
        - **get_datasets.bash**: *script* que permite descargar los conjuntos de datos utilizados  en la evaluación y que es una modificación del presente en el toolkit [SentEval](https://github.com/facebookresearch/SentEval).
        - **tokenizer.vec**
    - **embedding/**
        - **fasttext/get_fasttext_embeddings.bash**: *script* que descarga el conjunto de de vectores de palabras computado con FastText utilizado.
        - **gloVe/**
            - **2word2vec.py**: transforma el conjunto de vectores de GloVe al formato Word2Vec.
            - **get_glove_embeddings.bash**: *script* que descarga el conjunto de de vectores de palabras computado con GloVe utilizado.
        - **word2vec/get_word2vec_embeddings.bash**: *script* que descarga el conjunto de de vectores de palabras computado con Word2Vec utilizado.    
    - **frequencies.tsv**


- **.gitignore**

- **LICENSE**

- **SENTEVAL_LICENSE**: licencia del toolkit [SentEval](https://github.com/facebookresearch/SentEval) desarollado por Facebook.

- **evaluation.ipynb**: fichero de Jupyter Notebook en el que se desarrolla la evaluación realizada.

- **load.py**: contiene un conjunto de funciones para cargar y preprocesar los diferentes conjuntos de datos utilizados. El código está basado en el que se puede encontrar en el toolkit [SentEval](https://github.com/facebookresearch/SentEval).

- **methods.py**: contiene las funciones que implementan los tres métodos evaluados para calcular la similiridad semántica entre dos pares de oraciones: media (average), *Smooth Inverse Frequency* (SIF), y *Word Mover's distance*.

- **utils.py**: contiene algunas funciones de utilidad como para preprocesar las oraciones y evaluar los resultados de los diferentes métodos.


<hr>



# Comparison of pre-trained Word2Vec, GloVe and FastText vectors to measure semantic similarity between sentence pairs

- **data/**
    - **datatsets/**
    - **get_datasets.bash**: script that downloads the data sets used in the evaluation and which is a modification of the one present in the [SentEval](https://github.com/facebookresearch/SentEval) toolkit .
    - **tokenizer.vec**
    - **embedding/**
        - **fasttext/get_fasttext_embeddings.bash**: script that downloads the set of word vectors computed with the FastText used.
        - **gloVe/**
            - **2word2vec.py**: transforms the GloVe vector set to Word2Vec format.
            - **get_glove_embeddings.bash**: script that downloads the GloVe word embeddings set used.
    - **word2vec/get_word2vec_embeddings.bash**: script that downloads the Word2Vec word embeddings set used.
    - **frequencies.tsv**

- **.gitignore**

- **LICENSE**

- **SENTEVAL_LICENSE**: license of the [SentEval](https://github.com/facebookresearch/SentEval) toolkit  developed by Facebook.

- **evaluation.ipynb**: Jupyter Notebook file in which the evaluation carried out is developed.

- **load.py**: contains a set of functions to load and preprocess the different data sets used. The code is based on what can be found in the [SentEval](https://github.com/facebookresearch/SentEval) toolkit .

- **methods.py**: contains the functions that implement the three evaluated methods to calculate the semantic similarity between two pairs of sentences: average, Smooth Inverse Frequency(SIF), and Word Mover's distance.
- **utils.py**: contains some utility functions to preprocess sentences and evaluate the results of the different methods.


