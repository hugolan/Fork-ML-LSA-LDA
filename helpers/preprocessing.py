import random
import numpy as np
import pandas as pd
import re 

# Spacy
import spacy
from spacy.lang.fr.examples import sentences 
nlp = spacy.load("fr_core_news_sm")

# NLTK
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Evaluation
from gensim.models.coherencemodel import CoherenceModel


# gensim
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases
from gensim.corpora import Dictionary

# Use stopwords from two sources
nltk_stopwords = stopwords.words('french')
spacy_stopwords = spacy.lang.fr.stop_words.STOP_WORDS
stopwords = list(set(nltk_stopwords) | spacy_stopwords)

# Load data (i.e. articles) from a csv files and apply some pre-processing (sort + dates)
def load(filepath):
    """
    Loads dataset and sorts it by date
    """
    df = pd.read_csv(filepath, sep=";")
    df.date = pd.to_datetime(df.date)
    df = df.sort_values(by = "date")
    df = df.reset_index(drop=True)

    return df

# See data_cleanup.ipynb for some examples of why those filters
def cleanup(df):
    """
    Removes some irrelevant articles
    :param df: dataframe of articles
    :return: dataframe with rows removedd
    """
    # Remove null values
    df = df[df['content'].notna()]

    # Convert dates to datetime objects and sort by date
    df.date = pd.to_datetime(df.date)
    df = df.sort_values(by = "date")
    df = df.reset_index(drop=True)

    # specific strings from articles that were manually observed to be unrelated
    to_remove = ['CINÉMA Apollo 1 Faubourg', 'CINÉMA Eden Rue de la', 'NOUS RECHERCHONS', 'Ciné NEUCHÂTEL', \
             'Ciné LA CHAUX-DE-FONDS', 'AGENDA', 'APOLLO', 'ANIMATIONS', 'RATIONALISATION DES TRANSFERTS', \
             'REALITE', 'INTÉGRATION DU NÉGOCE DES OPTIONS', 'SOFFEX', 'www . hbo', 'Mythologies Primitive', \
             'GENEVE Marché', 'Morat-Central', 'GENEVE Reprise', 'monnaie plastique', 'La BCV', 'aint-Valentin', \
             'GENÈVE Marché', 'ÉCOLE DE NATATION', 'SOCIETES RÉSULTATS', 'SOCIETES SUISSE TRÉFILERIES', 'SOCIÉTÉS', \
             'Stock Exchange', 'CONGÉLATEUR BAHUT', 'USS Une', 'MARCHÉS BOURSIERS', 'TTVj', 'Bourse électronique', \
             'IAIIBERTé VIE', 'Mythologies Primitive', 'PASSONS A LA PRATIQUE', 'OCÉAN INDIEN', 'oeféeto', \
            'D M emain', 'LESBONS PLANS NEUCHÂTEL', 'NEUCHÂTEL CONCERT', 'FABULEUX PALAIS']
    df = df[~df.content.str.contains('|'.join(to_remove))]
    df = df[~(df.content.str.contains("bourse") & df.content.str.contains("banque"))]
    
    # articles with many telephone numbers were advertisements
    df = df[df.content.str.count('026') < 3]
    df = df[df.content.str.count('032') < 3]
    df = df[df.content.str.count('00') < 6]
    
    # articles with many ° are wether bulletins
    df = df[df.content.str.count('°') < 3]
    
    df = df.reset_index(drop=True)
    
    # Some regular expressions to remove telephone numbers and other combinations
    test = df.content.apply(lambda x: re.search('\d\d\s\.\s\d\d', x))
    test2 = df.content.apply(lambda x: re.search('\~\~', x))

    tests = [test, test2]

    for t in tests:
        none = []
        for i in range(len(t)):
            none.append(t[i] is None)
        none = pd.Series(none)

        none_index = np.where(~none)

        df = df.drop(none_index[0])
        df = df.reset_index(drop=True)
    
    # Articles with multiple phone numbers are advertisements, remove them
    test3 = df.content.apply(lambda x: len(re.findall('\d\d\d\s\d\d\s\d\d', x)))
    df = df.drop(test3[test3 > 2].index)
    df = df.reset_index(drop=True)
    
    # Remove really short articles
    df = df[df.content.str.len() > 300]
    df = df.reset_index(drop=True)


    return df


def remove_stopwords(data):
    """
    Use gensim simple_preprocess to tokenize and remove accents
    :param data: documents 
    :return: list of words
    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in data]

    
def remove_keywords(content):
    """
    Remove certain keywords that might dominate results
    :param data: documents
    :return: list of words without certain keywords
    """
    split = content.split()
    final = [word for word in split if word.lower() not in ['jeu', 'vidéo', 'jeu video', 'jeu vidéo', 'video', 'suisse', 'nintendo', 'sega', 'atari', 'playstation', 'xbox']]
    final = ' '.join(final)
    return final


def spacy_prep(data):
    """
    Takes a dataframe column of articles and preprocesses it (spacy's full pipeline)
    :param data: articles
    :return: list of processed documents
    """
    data = [[word for word in simple_preprocess(str(doc), deacc=True)] for doc in data]
    data = [' '.join(doc) for doc in data]
    data = pd.Series(data)
    
    processed_docs = list()
    for doc in nlp.pipe(data, n_threads=5, batch_size=100):

        ents = doc.ents  # Named entities
        
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']

        # Remove punctuation and numbers
        doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and token.pos_ in allowed_postags]

        # Remove stopwords and remove words less than length 3
        doc = [token for token in doc if token not in stopwords and len(token) > 2]

        # Add entities that are compounds of more than one word
        doc.extend([str(entity) for entity in ents if len(entity) > 1])
        

        processed_docs.append(doc)
    docs = processed_docs
    del processed_docs
    
    # Add bigrams to docs that appear 5 times or more
    bigram = Phrases(docs, min_count=5)

    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                docs[idx].append(token)
    
    return docs


def create_corpus(docs, min_wordcount, max_freq):
    """
    Creates a dictionary and corpus from the documents
    :param docs: documents
    :param min_wordcount: minimum number of documents a word needs to appear in to be kept
    :param max_freq: maximum percentage of documents a word should appear in the be kept
    :return: dictionary and corpus
    """
    dictionary = Dictionary(docs)

    # Filter out words that are in less than min_wordcount documents and that appear in more than max_freq documents
    dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)
    dictionary.compactify()

    # Bag-of-words representation of the documents
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    
    return dictionary, corpus

def create_test_corpus(dictionary, test):
    """
    Create a corpus for the test set using the dictionary of the training set
    :param dictionary: dictionary of the training set
    :param test: test set
    :return: the corpus for the test set 
    """
    corpus = [dictionary.doc2bow(doc) for doc in test]
    
    return corpus


def train_test_split(data, prop_train):
    """
    Splits the data for training and testing
    :param data: the full set of documents
    :param prop_train: the proportion of the data to assign to training
    :return: the training set and test set
    """
    training_size = int(float(len(data)) * prop_train)
    train_index = random.sample(range(0, len(data)-1), training_size)

    content=np.array(data)
    mask=np.full(len(data),False,dtype=bool)
    mask[train_index]=True
    test=pd.Series(content[~mask])
    train=pd.Series(content[mask])
    
    return train, test

    
    
    