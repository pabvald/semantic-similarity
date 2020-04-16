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
- **methods.py**: contiene las funciones que implementan los tres métodos evaluados para calcular la similiridad semántica entre dos pares de oraciones: media (average), *Smooth Inverse Frequency*(SIF), y *Word Mover's distance*.
- **utils.py**: contiene algunas funciones de utilidad como para preprocesar las oraciones y evaluar los resultados de los diferentes métodos.


<hr>



# Comparison of pre-trained Word2Vec, GloVe and FastText vectors to measure semantic similarity between sentence pairs



