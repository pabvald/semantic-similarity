## Content

- `data/`
    - `datatsets/`
        - `get_datasets.bash`: *script* to download the datasets used in the evaluation, which is a modification of the one provided in the [SentEval](https://github.com/facebookresearch/SentEval) toolkit.
        - `tokenizer.vec`
    - `embedding/`
        - `fasttext/get_fasttext_embeddings.bash`: script that downloads the set of word vectors computed with the FastText used.
        - `gloVe/`
            - `2word2vec.py`: transforms the GloVe vector set to Word2Vec format.
            - `get_glove_embeddings.bash`: script that downloads the GloVe word embeddings set used.
    - `word2vec/get_word2vec_embeddings.bash`: script that downloads the Word2Vec word embeddings set used.
    - `frequencies.tsv`

- `evaluation.ipynb`: Jupyter Notebook file in which the evaluation carried out is developed.

- `load.py`: contains a set of functions to load and preprocess the different data sets used. The code is based on what can be found in the [SentEval]To run the evaluation code, contained in the Jupyter Notebook file [evaluation.ipynb](./evaluation.ipynb), you can follow the following steps:

## Evaluation 

### 1. Installing Python3.7 and its virtual environment tool
First, install Python3.7 and the virtual environment tool: 
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7 
sudo apt install python3.7-venv
```

### 2. Creating and Activating a Python3.7 Virtual Environment
Second, create a Python3.7 virtual environment inside this repository:

```
python3.7 -m venv .venv 
```
and activate it: 
```
source .venv/bin/activate
```

### 3. Installing the dependencies 
Once the virtual environment is activated, install the dependencies using the following command:
```
pip install -r requirements.txt 
```

### 4. Downloading vector sets 
Note that in order to reproduce the evaluation contained in the [evaluation.ipynb](./evaluation.ipynb) file, you must first download the Word2Vec, GloVe and FastText word vector sets. Each of these sets is of considerable size and may take several minutes to download. 

#### 4.1. Downloading the Word2Vec set 
With this repository (*semantic_similarity/*) being the current directory, run the following commands:

```
cd data/embedding/word2vec 
chmod +x get_word2vec_embeddings.bash 
./get_word2vec_embeddings.bash
```

#### 4.2. Downloading the GloVe set 
With this repository (*semantic_similarity/*) being the current directory, run the following commands:

```
cd data/embedding/glove 
chmod +x get_glove_embeddings.bash 
./get_glove_embeddings.bash
python 2word2vec.py
```

#### 4.3. Downloading the FastText set 
With this repository (*semantic_similarity/*) being the current directory, run the following commands:

```
cd data/embedding/fasttext 
chmod +x get_fasttext_embeddings.bash 
./get_fasttext_embeddings.bash
```

### 5. Downloading the datasets
It is also necessary to download the datasets. For them, this repository (*semantic_similarity/*) being the current directory, run the following commands:
```
cd data/datasets
sudo chmod +x get_datasets.bash 
./get_datasets.bash
```

### 6. Starting Jupyter Notebook 
Run Jupyter Notebook and access the *evaluation.ipynb* file. To run Jupyter Notebook, execute the following command:
```
jupyter-notebook
```

Once you have finished using Jupyter Notebook, in the terminal where you executed the previous command, use `Ctrl + C` to end the execution of Jupyter Notebook. Finally, disable the virtual environment using the following command:
``` 
deactivate 
```

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
