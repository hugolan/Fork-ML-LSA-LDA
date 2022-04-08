from collections import defaultdict
from datetime import datetime
import math
from operator import itemgetter
import os
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LDA
from gensim.models import LdaMulticore

# LSA
from gensim.models import LsiModel

# Evaluation
from gensim.models.coherencemodel import CoherenceModel



# LSA
def lsa(dictionary, corpus, num_topics, num_words, power_iters):
    """
    Returns a latent semantic analysis model for a chosen number of topics and prints a given number of words for each topic
    :param dictionary: the dictionary of words
    :param corpus: bag of words representation of the the documents
    :param num_topics: the number of topics to detect
    :param num_words: the number of words to print for each topic
    :param power_iters: number of iterations
    :return: the lsa model object
    """
    lsamodel = LsiModel(corpus, num_topics=num_topics, id2word = dictionary, power_iters=power_iters)  # train model
    print(lsamodel.print_topics(num_topics=num_topics, num_words=num_words))
    return lsamodel


# LDAMulticore -- equivalent to LDAmodel just more efficient
def lda(dictionary, corpus, num_topics, workers, passes, random_state, alpha, eta):
    """
    Create a Latent Dirichlet Allocation model 
    :param dictionary: the dictionary of words
    :param corpus: bag of words representation of the the documents
    :param num_topics: the number of topics to detect
    :param workers: number of workers to be used for parallelization
    :param passes: number of times you go through the entire corpus
    :param random_state: seed for reproducibility
    :param num_words: the number of words to print for each topic
    :param alpha: controls the prior distribution over topic weights in each document
    :param eta: controls the the prior distribution over word weights in each topic
    :return: the lda model object
    """
    # Use default alpha and eta
    if ((alpha == None) & (eta == None)):
        model = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, workers=workers,
                    passes=passes, random_state=random_state)
    else:
        model = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, workers=workers,
                    passes=passes, random_state=random_state, alpha=alpha, eta=eta)
    
    return model


# Perplexity

def compute_perplexity(lda_model, corpus):
    """
    Compute perplexity for an lda_model and corresponding corpus
    :param lda_model: the trained lda model
    :param corpus: the corpus for the data
    :return: perplexity
    """
    return lda_model.log_perplexity(corpus)


# Coherence

def compute_coherence(model, data, corpus, dictionary, coherence_measure):
    """
    Computes the coherence for the given model
    :param data: the clean data (articles)
    :param corpus: bag of words representation of the the documents
    :param dictionary: the dictionary of words
    :param coherence_measure: one of the following: {'u_mass', 'c_v', 'c_uci', 'c_npmi'}
    :return: coherence value
    """
    coherence_model = CoherenceModel(model=model, texts=data, dictionary=dictionary, coherence=coherence_measure)
    coherence = coherence_model.get_coherence()
    return coherence


def compute_coherences_lsa(dictionary, corpus, data, start, stop, step, power_iters):
    """
    Compute coherence values for different numbers of topics
    :param dictionary: the dictionary of words
    :param corpus: bag of words representation of the the documents
    :param data: the clean data (articles)
    :param start: the min number of topics to look at 
    :param stop: the max number of topics to look at 
    :param step: step value for number of topics
    :param power_iters: number of iterations for LSA
    :return: the models and their corresponding coherence score
    """
    coherences = []
    models = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(corpus, num_topics=num_topics, id2word = dictionary, power_iters=power_iters)  # train model
        models.append(model)
        coherencemodel = CoherenceModel(model=model, texts=data, dictionary=dictionary, coherence='c_v')
        coherences.append(coherencemodel.get_coherence())
    return models, coherences


# See optimum number of topics
def plot_graph(data, dictionary, corpus, start, stop, step, power_iters):
    """
    Plots the number of topics vs the coherence value
    :param data: the clean data (articles)
    :param dictionary: the dictionary of words
    :param corpus: bag of words representation of the the documents
    :param start: the min number of topics to look at 
    :param stop: the max number of topics to look at 
    :param step: step value for number of topics
    :param power_iters: number of iterations for LSA
    """
    models, coherences = compute_coherences_lsa(dictionary, corpus, data, start, stop, step, power_iters)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherences)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence values"), loc='best')
    plt.show()
    
def view_topics(topic_keys, num_words):
    """
    View the words for each topic
    :param topic_keys: the topics generated from the model
    :param num_words: the number of words to look at for each topic
    """
    for i, t in enumerate(topic_keys):
        print(i, '\t', ' '.join(t[:num_words]))
        
def get_top_docs(training_data, topic_distributions, num_docs, topic_index):
    """
    View the top documents for each topic
    :param training_data: the data
    :param topic_distributions: probability that a document belongs to a certain topic
    :param num_docs: the number of documents to show for each topic
    :param topic_index: the topic to look at
    """
    for prob, doc in lmw.get_top_docs(training_data, topic_distributions, topic_index=topic_index, n=num_docs):
        print(round(prob, 4), doc)
        print()

def get_top_words(topic_word_probability_dict, num_words):
    """
    View the top words for each topic
    :param topic_word_probability_dict: dictionary with the probablility of each word for each topic
    :param num_words: the number of words to view
    """
    for topic, word_probability_dict in topic_word_probability_dict.items():
        print('Topic', topic)
        for word, prob in sorted(word_probability_dict.items(), key=lambda x: x[1], reverse=True)[:num_words]:
            print(round(prob, 4), '\t', word)
        print()

