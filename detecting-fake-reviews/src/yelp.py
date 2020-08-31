# Python Script for Parsing the hauyi dataset
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np

import bow
import classifiers
from sklearn.model_selection import train_test_split

DEBUG = False

'''Simple class to store metadata information'''
class YelpMetadata:
    def __init__(self, date, review_id, reviewer_id, product_id, label):
        self.date        = date
        self.review_id   = review_id
        self.reviewer_id = reviewer_id
        self.product_id  = product_id
        self.label       = 'd' if label == 'Y' else 't'

'''Simple class that serves as a wrapper for metadata and raw review text'''
class YelpReview:
    def __init__(self, text: str, metadata: YelpMetadata):
        self.text = text
        self.metadata = metadata

def get_yelp_chi_hotel_reviews():
    print('INSIDE HOTEL')
    hotel_data        = []
    hotelMetadataList = []
    hotelReviewsList  = []

    yelp_dataset_path = '../datasets/yelp/'
    yelp_chi_path = 'YelpChi/'
    metadata_file_path = yelp_dataset_path + yelp_chi_path + 'output_meta_yelpHotelData_NRYRcleaned.txt'
    reviews_file_path  = yelp_dataset_path + yelp_chi_path + 'output_review_yelpHotelData_NRYRcleaned.txt'

    '''Read metadata and fill hotelMetadataList with YelpMetadata objects'''
    with open(metadata_file_path) as metadata_file:
        line = metadata_file.readline()
        while line:
            cols = line.split(' ')
            metadata = YelpMetadata(cols[0], cols[1], cols[2], cols[3], cols[4])
            hotelMetadataList.append(metadata)
            line = metadata_file.readline()


    with open(reviews_file_path) as review_file:
        line = review_file.readline()
        while line:
            hotelReviewsList.append(line)
            line = review_file.readline()

    assert(len(hotelMetadataList) == len(hotelReviewsList))
    n = len(hotelMetadataList)

    for i in range(n):
        hotelMetadata   = hotelMetadataList[i]
        hotelReviewText = hotelReviewsList[i]
        yelpReview      = YelpReview(hotelReviewText, hotelMetadata)
        hotel_data.append(yelpReview)
        if DEBUG:
            print('-------------------------------------------------------------------------------')
            print('Review [{i}]'.format(i=i))
            print(' * Metadata:')
            print('    - Date:        {x}'.format(x=yelpReview.metadata.date))
            print('    - Review ID:   {x}'.format(x=yelpReview.metadata.review_id))
            print('    - Reviewer ID: {x}'.format(x=yelpReview.metadata.reviewer_id))
            print('    - Product ID:  {x}'.format(x=yelpReview.metadata.product_id))
            print('    - Label:       {x}\n'.format(x=yelpReview.metadata.label))
            print('\nReview Text: {}\n\n\n'.format(yelpReview.text))

    return hotel_data

def get_yelp_chi_restaurant_reviews():
    print('INSIDE RESTAURANT')
    restaurant_data        = []
    restaurantMetadataList = []
    restaurantReviewsList  = []

    yelp_dataset_path = '../datasets/yelp/'
    yelp_chi_path = 'YelpChi/'
    metadata_file_path = yelp_dataset_path + yelp_chi_path + 'output_meta_yelpResData_NRYRcleaned.txt'
    reviews_file_path  = yelp_dataset_path + yelp_chi_path + 'output_review_yelpResData_NRYRcleaned.txt'

    '''Read metadata and fill restaurantMetadataList with YelpMetadata objects'''
    with open(metadata_file_path) as metadata_file:
        line = metadata_file.readline()
        while line:
            cols = line.split(' ')
            metadata = YelpMetadata(cols[0], cols[1], cols[2], cols[3], cols[4])
            restaurantMetadataList.append(metadata)
            line = metadata_file.readline()

    with open(reviews_file_path) as review_file:
        line = review_file.readline()
        while line:
            restaurantReviewsList.append(line)
            line = review_file.readline()

    assert(len(restaurantMetadataList) == len(restaurantReviewsList))
    n = len(restaurantMetadataList)

    for i in range(n):
        restaurantMetadata   = restaurantMetadataList[i]
        restaurantReviewText = restaurantReviewsList[i]
        yelpReview           = YelpReview(restaurantReviewText, restaurantMetadata)
        restaurant_data.append(yelpReview)
        if DEBUG:
            print('-------------------------------------------------------------------------------')
            print('Review [{i}]'.format(i=i))
            print(' * Metadata:')
            print('    - Date:        {x}'.format(x=yelpReview.metadata.date))
            print('    - Review ID:   {x}'.format(x=yelpReview.metadata.review_id))
            print('    - Reviewer ID: {x}'.format(x=yelpReview.metadata.reviewer_id))
            print('    - Product ID:  {x}'.format(x=yelpReview.metadata.product_id))
            print('    - Label:       {x}\n'.format(x=yelpReview.metadata.label))
            print('\nReview Text: {}\n\n\n'.format(yelpReview.text))
    return restaurant_data


def get_chi_reviews():
    chi_hotel_yelp_reviews = get_yelp_chi_hotel_reviews()

    chi_restaurant_yelp_reviews = get_yelp_chi_restaurant_reviews()
    chi_yelp_reviews = chi_hotel_yelp_reviews + chi_restaurant_yelp_reviews
    reviews = []
    scores = []
    # length_of_reviews = []

    for yelp_review in chi_yelp_reviews:
        reviews.append(yelp_review.text)

        # length_of_reviews.append(len(yelp_review.text))

        if yelp_review.metadata.label == 'd':
            scores.append(0)
        else:
            scores.append(1)

    #        scores.append(yelp_review.metadata.label)
    return reviews, scores

if __name__ == '__main__':
    print('yelp')
    reviews, scores = get_chi_reviews()
    print(len(reviews), len(scores))

    bow, vec = bow.bag_of_words(reviews)

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