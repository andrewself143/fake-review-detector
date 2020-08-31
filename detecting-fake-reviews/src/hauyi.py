# Python Script for Parsing the hauyi dataset
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np

import bow
import classifiers
from sklearn.model_selection import train_test_split

def gather_stopwords():
    print('Gathering stop words')
    stop_words = []
    file_name = '../datasets/data-hauyi/stopwords.txt'
    for line in open(file_name):
        line = line.strip()
        # print(line)
        stop_words.append(line)
    # print(stop_words)
    return stop_words

def read_chinese():
    print('Reading Chinese')
    # save reviews and laebels to an arry
    # file_name = '../datasets/data-hauyi/ICDM_REVIEWS_TO_RELEASE_encoding=utf-8.csv'
    file_name = '../datasets/data-hauyi/reviews.txt'


    labels = []
    reviews = []

    count = 0
    for line in open(file_name):
        # print('HERE')
        # count = count
        count = count + 1
        if count == 1:
            continue

        # line = line.split(',', maxsplit=5) # max split equals 5 so as to not split
        line = line.split(' ', maxsplit=1) # max split equals 5 so as to not split

        # print(line)
        # label = line[1]
        # review = line[5]
        label = line[0]
        review = line[1]
        # print('label')
        # print('review')
        # if label == '+':
        if label == '0':
            labels.append(0)
        else:
            labels.append(1)
        reviews.append(review)

        if count > 9000:
            break
        # print(count)

    # print(len(labels), len(reviews))
    # print(labels)
    return reviews, labels

def segment(labels, reviews):
    '''ONLY NEEDS TO BE RUN ONCE'''
    # save all segmented reviews to a file
    segmented = []
    # seg = StanfordSegmenter('../../datasets/data-hauyi/stanford-segmenter-2018-10-16')
    os.environ["STANFORD_SEGMENTER"] = '../datasets/data-hauyi/stanford-segmenter-2018-10-16'
    seg = StanfordSegmenter('../datasets/data-hauyi/stanford-segmenter-2018-10-16/stanford-segmenter-3.9.2.jar')
    seg.default_config('zh',)
    count = 0

    file_out = open('reviews.txt','a+')

    for i in range(len(reviews)):
        # print(i)
        s = seg.segment(reviews[i])
        l = labels[i]
        # print(s)
        line = str(l) + ' ' + s
        file_out.write(line)
        segmented.append(s)
        # print('Tokenize: ')
        # print(seg.tokenize(s))
        count = count + 1
        # if count > 5:
        #     break
        print('Count: ', count)

    return(segmented)

def chinese_BOW(reviews, stop):
    # vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", min_df=0.01)
    # vectorizer = CountVectorizer()
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop, min_df=0.01) # create vectorizer
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer


if __name__ == '__main__':
    print("hauyi.py")
    stopwords = gather_stopwords()

    reviews, labels = read_chinese()
    # print(len(labels), len(reviews))

    bow, vec = bow.chinese_BOW(reviews, stopwords)
    # print(bow)

    train_x, test_x, train_y, test_y = train_test_split(bow, labels, test_size=0.25, random_state=42)

    classifier = classifiers.logistic_regression(train_x, train_y)

    train_predictions = classifier.predict(train_x)  # Training
    train_accuracy = metrics.accuracy_score(train_y, train_predictions)
    class_probabilities_train = classifier.predict_proba(train_x)
    train_auc_score = metrics.roc_auc_score(train_y, class_probabilities_train[:, 1]);
    print('\nTraining:')
    print(' accuracy:', format(100 * train_accuracy, '.2f'))
    print(' AUC value:', format(100 * train_auc_score, '.2f'))

    test_predictions = classifier.predict(test_x)  # Test
    test_accuracy = metrics.accuracy_score(test_y, test_predictions)
    class_probabilities_test = classifier.predict_proba(test_x)
    test_auc_score = metrics.roc_auc_score(test_y, class_probabilities_test[:, 1]);
    print('\nTesting:')
    print(' accuracy:', format(100 * test_accuracy, '.2f'))
    print(' AUC value:', format(100 * test_auc_score, '.2f'))