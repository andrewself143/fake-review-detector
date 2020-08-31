from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np
import gensim
from gensim.models import Doc2Vec
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import classifiers
import yelp
import op_spam
import hauyi
import bow
from sklearn import metrics



def get_corpus(reviews: [str], scores: [int], lang):
    '''
    Iterate over the reviews and corresponding scores and create a TaggedDocument
    object for each pair. These TaggedDocument objects make it easier to create Training
    and Testing matrices.
    '''
    if lang == 'en':
        stoplist = stopwords.words('english')
    else:
        stoplist = hauyi.gather_stopwords()
    review_tokens = []
    for review in reviews:
        review_tokens.append([word for word in review.lower().split() if word not in stoplist])
    for i, text in enumerate(review_tokens):
        yield gensim.models.doc2vec.TaggedDocument(text, [scores[i]])


def add_unique_labels(train_regressors):
    '''Go through the labels vector and give a unique ID to each label.'''
    Y = np.asarray(train_regressors)
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(Y)
    train_y = labelEncoder.transform(Y)
    return train_y


def create_doc2vec_model(train_corpus):
    model = Doc2Vec(window=100, dm=1, vector_size=50, min_count=2)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def get_train_lists(model, train_targets, train_regressors):
    X = []
    for i in range(len(train_targets)):
        X.append(model.infer_vector(train_targets[i]))

    train_x = np.asarray(X)
    train_y = add_unique_labels(train_regressors)
    return train_x, train_y


def get_test_lists(model, test_targets, test_regressors):
    test_list = []
    for i in range(len(test_targets)):
        test_list.append(model.infer_vector(test_targets[i]))

    test_x = np.asarray(test_list)
    test_y = add_unique_labels(test_regressors)
    return test_x, test_y

def run_classifiers_with_doc2vec(reviews, scores, lang='en'):
    '''Corpus should be an array of TaggedDocument objects.'''
    corpus = list(get_corpus(reviews, scores, lang))[:20000]
    train_corpus, test_corpus = train_test_split(corpus, test_size=0.25, random_state=42)

    doc2vec_model = create_doc2vec_model(train_corpus)
    train_targets, train_regressors = zip(*[(doc.words, doc.tags[0]) for doc in train_corpus])
    test_targets, test_regressors = zip(*[(doc.words, doc.tags[0]) for doc in test_corpus])

    '''
    For every review, we apply doc2vec_model.infer_vector(review). This creates
    a feature vector for every document (in our case, review) in the corpus.
    '''
    train_x, train_y = get_train_lists(doc2vec_model, train_targets, train_regressors)
    test_x,  test_y  = get_test_lists(doc2vec_model, test_targets, test_regressors)

    classifier = classifiers.logistic_regression(train_x, train_y)
    # classifier = classifier_func(train_x, train_y)
    # return logistic_reg
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


if __name__ == '__main__':
    print('embeddings')
    # reviews, scores = yelp.get_chi_reviews()
    reviews, scores = op_spam.parse_op_spam()
    # reviews, scores = hauyi.read_chinese()
    print(len(reviews), len(scores))

    # bow, vec = bow.bag_of_words(reviews)

    # run_classifiers_with_doc2vec(reviews, scores, lang='zh')
    run_classifiers_with_doc2vec(reviews, scores, lang='en')

