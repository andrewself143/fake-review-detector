import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np


def bag_of_words(reviews):
    '''
    Creating a bag of words by counting the number of times each word appears in a document.
    This is possible using CountVectorizer
    '''
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", min_df=0.01)

    '''Create a sparse BOW array from 'text' using vectorizer'''
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer

def chinese_BOW(reviews, stop):
    # vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", min_df=0.01)
    # vectorizer = CountVectorizer()
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop, min_df=0.01) # create vectorizer
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer

if __name__ == '__main__':
    print('BOW')