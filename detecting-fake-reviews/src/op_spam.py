# Python Script for Parsing the op_spam dataset
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np

import bow
import classifiers
from sklearn.model_selection import train_test_split

def parse_op_spam():
    """get text of review and t/f value"""
    reviews = list()
    scores = list()
    # length_of_reviews = list()

    # negative_polarity directory
    file_path_negative_polarity = "../datasets/op_spam_v1.4/negative_polarity/"
    files_in_directory_negative_polarity = os.listdir(file_path_negative_polarity)

    for file_name in files_in_directory_negative_polarity:

        # print('file_name: ', file_name)
        file_flag = file_name[0]
        file_path = file_path_negative_polarity + file_name
        file_open = open(file_path)
        review = file_open.readline()
        if file_flag == "d":
            scores.append(0)
        else:
            scores.append(1)

        reviews.append(review)
        # length_of_reviews.append(len(review))

    # positive_polarity directory
    file_path_positive_polarity = "../datasets/op_spam_v1.4/positive_polarity/"
    files_in_directory_positive_polarity = os.listdir(file_path_positive_polarity)

    for file_name in files_in_directory_positive_polarity:

        file_flag = file_name[0]
        file_path = file_path_positive_polarity + file_name
        file_open = open(file_path)
        # print('file_name: ', file_name)
        review = file_open.readline()

        if file_flag == "d":
            scores.append(0)
        else:
            scores.append(1)

        reviews.append(review)
        # length_of_reviews.append(len(review))

    return reviews, scores #, length_of_reviews

if __name__ == '__main__':
    print("OP_SPAM.py")
    reviews, scores = parse_op_spam()
    print(len(reviews), len(scores))
    bow, vec = bow.bag_of_words(reviews)
    # print(bow)

    train_x, test_x, train_y, test_y = train_test_split(bow, scores, test_size=0.25, random_state=42)

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
