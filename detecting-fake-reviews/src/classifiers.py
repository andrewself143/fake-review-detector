from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# import functions

import bow
import hauyi
import yelp
import op_spam
import bow
import embeddings


def logistic_regression(X, Y, test_fraction=0.25):
    classifier = linear_model.LogisticRegression(penalty='l2', fit_intercept=True)
    classifier.fit(X, Y)
    return classifier


def naive_bayes(X, Y, test_fraction=0.25):
    classifier = MultinomialNB()
    classifier.fit(X, Y)
    return classifier


def knearest_neighbors(X, Y, test_fraction=0.25):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X, Y)
    return classifier


def decision_trees(X, Y, test_fraction=0.25):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X, Y)
    return classifier


def random_forest(X, Y, test_fraction=0.25):
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X, Y)
    return classifier

def test_classifier(model, x, y):
    predictions = model.predict(x)  # Training
    accuracy = metrics.accuracy_score(y, predictions)
    class_probabilities = model.predict_proba(x)
    auc_score = metrics.roc_auc_score(y, class_probabilities[:, 1]);
    print('\nResults:')
    print(' accuracy:', format(100 * accuracy, '.2f'))
    print(' AUC value:', format(100 * auc_score, '.2f'))


if __name__ == '__main__':

    models = [['logistic_regression',logistic_regression],
              ['naive_bayes',naive_bayes],
              ['knearest_neighbors',knearest_neighbors],
              ['decision_trees',decision_trees],
              ['random_forest',random_forest]]

    for x in models:
        name, func = x
        print(name)
        print(func)

    # datasets = [['chinese',hauyi.read_chinese],
    #             ['yelp',yelp.get_chi_reviews] ,
    #             ['op_spam',op_spam.parse_op_spam]]

    datasets = [['yelp', yelp.get_chi_reviews],
                ['op_spam', op_spam.parse_op_spam]]

    for x in datasets:
        name, func = x
        print(name)
        print(func)

    for x in datasets:
        name, func = x
        print('----------', name, '-----------')
        reviews, labels = func()

        # if name == 'chinese':
        #     print('entered chinese')
        #     stopwords = hauyi.gather_stopwords()
        #     bow_data, vec = bow.chinese_BOW(reviews, stopwords)
        # else:
        #     bow_data, vec = bow.bag_of_words(reviews)

        # train_x, test_x, train_y, test_y = train_test_split(bow_data, labels, test_size=0.25, random_state=42)
        embeddings.run_classifiers_with_doc2vec(reviews, labels, lang='en')
        # for x in models:
        #     name, func = x
        #     print('-------', name)
        #     embeddings.run_classifiers_with_doc2vec(func, reviews, labels, lang='en')
            # classifier = func(train_x, train_y)
            # print('TRAIN DATA')
            # test_classifier(classifier, train_x, train_y)
            # print('TEST DATA')
            # test_classifier(classifier, test_x, test_y)



        # print(func)

    #BOW
    # for x in datasets:
    #     for x in classifiers
    #         test_classifier()

    #Embdeddings
