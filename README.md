# Comparativa de vectores pre-entrenados de Word2Vec, GloVe y FastText para medir la similaridad semántica entre pares de oraciones (English below)

Parte del Trabajo de Fin de Grado **"Asistentes virtuales: estado del arte y desarrollo de un prototipo"** realizado por D. Pablo Valdunciel Sánchez.
 
## Estructura del repositorio

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



## Evaluación
Para ver la evaluación, acceda al fichero [evaluation.pdf](./evaluation.pdf). En el caso de que desee ejecutar el código de la evaluación, contenido en el fichero de Jupyter Notebook [evaluation.ipynb](./evaluation.ipynb), puede seguir los siguientes pasos:

### 1. Instalación de  Python3.7 y su herramienta de entornos virtuales
En primer lugar, instale Python3.7 y la herramienta de entornos virtuales: 
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7 
sudo apt install python3.7-venv
```

### 2. Creación y activación de un entorno virtual de Python3.7
En segundo lugar, cree un entorno virtual de Python3.7 dentro de este repositorio:

```
python3.7 -m venv .venv 
```
y actívelo: 
```
source .venv/bin/activate
```

### 3. Instalación de las dependencias 
Una vez activado el entorno virtual, instale las dependencias mediante el siguiente comando:
```
pip install -r requirements.txt 
```
### 4. Descarga de los conjuntos de vectores 
Tenga en cuenta que para reproducir la evaluación contenida en el fichero [evaluation.ipynb](./evaluation.ipynb) debe descargar previamente los conjuntos de vectores de palabras de Word2Vec, GloVe y FastText. Cada uno de estos conjuntos tiene un tamaño considerable y su descarga puede llevar varios minutos. 

#### 4.1. Descarga del conjunto Word2Vec 
Siendo este repositiorio (*semantic_similarity/*) el directorio actual, ejecute los siguientes comandos:

```
cd data/embedding/word2vec 
chmod +x get_word2vec_embeddings.bash 
./get_word2vec_embeddings.bash
```

#### 4.2. Descarga del conjunto GloVe 
Siendo este repositiorio (*semantic_similarity/*) el directorio actual, ejecute los siguientes comandos:

```
cd data/embedding/glove 
chmod +x get_glove_embeddings.bash 
./get_glove_embeddings.bash
python 2word2vec.py
```

#### 4.3. Descarga del conjunto FastText 
Siendo este repositiorio (*semantic_similarity/*) el directorio actual, ejecute los siguientes comandos:

```
cd data/embedding/fasttext 
chmod +x get_fasttext_embeddings.bash 
./get_fasttext_embeddings.bash
```


### 5. Descarga de los conjuntos de datos
También es necesario descargar los conjuntos de datos. Para ellos, siendo este repositiorio (*semantic_similarity/*) el directorio actual, ejecute los siguientes comandos:
```
cd data/datasets
sudo chmod +x get_datasets.bash 
./get_datasets.bash
```

### 6. Inicio de Jupyter Notebook 
Ejecute Jupyter Notebook y acceda al fichero *evaluation.ipynb*. Para ejecutar Jupyter Notebook, ejecute el siguiente comando:
```
jupyter-notebook
```

Una vez termine de utilizar Jupyter Notebook, en el terminal en el que ejecutó el anterior comando, utilice *Ctrl + C* para finalizar la ejecución de Jupyter Notebook. Por último, desactive el entorno virtual mediante el siguiente comando:
``` 
deactivate 
```

## Dependencias 
```
gensim==3.8.2
jupyter==1.0.0
notebook==6.0.3
numpy==1.18.3
Orange3==3.25.0
pandas==1.0.3
sklearn==0.0
spacy==2.2.4
```

<hr>



# Comparativa de vectores pre-entrenados de Word2Vec, GloVe y FastText para medir la similaridad semántica entre pares de oraciones


## Repository structure

- **data/**
    - **datatsets/**
        - **get_datasets.bash**: *script* que permite descargar los conjuntos de datos utilizados  en la evaluación y que es una modificación del proporcionado en el toolkit [SentEval](https://github.com/facebookresearch/SentEval).
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


## Evaluation

[See evaluation](./evaluation.ipynb) (in Spanish)

## Dependencies
```
gensim==3.8.2
jupyter==1.0.0
notebook==6.0.3
numpy==1.18.3
Orange3==3.25.0
pandas==1.0.3
sklearn==0.0
spacy==2.2.4
```
