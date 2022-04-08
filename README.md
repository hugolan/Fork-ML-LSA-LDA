# ML Project 2: Unsupervised Topic Modeling of Video Game Related Articles

This repository contains the code for the ML4Science Project. In this project, we look at different ways of generating topics from a news corpus. 
The corpus contains Swiss articles related to video games from 1964 to 2018. 
Unsupervised models such as Latent Semantic Analysis and Latent Dirichlet Allocation are used to create topic representations.


## 1. Data
The dataset comes from the Impresso project. 
More details can be found [here](data/source.md).

## 2. Libraries/How to Run

Our code requires many libraries, all of them available through conda or pip:
- Conda `conda intall <library-name>`:
  - numpy
  - pandas
  - seaborn
  - matplotlib
  - gensim
  - notebook
  - nltk
  - vadersentiment
  - vadersentiment-fr
  - sklearn
  - tensorflow
- pip `pip install <library-name>`:
  - bertopic
- others:
  - `python -m spacy download fr_core_news_sm` 
  
All of the code can then be run as notebooks:
```
cd <path to the directory of this file>
conda activate # If conda not already active
jupyter notebook
```
Bertopic may cause problems in Jupyter, but can be run in Google Colab.

## 3. Repository Structure and Files
### helpers folder
`preprocessing.py`: loading the data, cleaning the data, pipeline for NLP, creating a dictionary and corpus

`models.py`: LSA and LDA models, evaluation metrics

`utilities.py`: sentiment analysis helpers

### models folder
* pretrained LDA model
* Brown clustering output

### data folder
* impresso3.csv: dataset

### coherence folder
* results from hyperparameter tuning that takes a significant amount of time to run

### Notebooks
`lda_tuning.ipynb` : Latent Dirichlet Allocation training, tuning, exploration, plotting of topics over time

`other_models.ipynb` : Latent Semantic Analysis (gensim), Brown Clustering

`sentiments.ipynb` : Sentiment Analysis

`Tf-idf.ipynb` : Exploration of the data by projecting it in lower dimensions to identify topics and clusters (LSA)

`Word2Vec.ipynb` : Example of the usage of Word2Vec for word embedding

`berttopicmodel.ipynb` : Exploration using BERTopic 

### Additional files

Cluster_axis_1_4.html is the output of an interactive plot from the notebook which allows to see point with assigned journals and years from the tf-idf notebook.
