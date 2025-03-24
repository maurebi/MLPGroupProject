# Machine Learning Project
# Nikki van Gurp, Ilse Kerkhove, Dertje Roggeveen & Marieke Schelhaas
import os
import re
import pandas as pd
import sklearn
import time
from baseline import *
from svm_classifier import SVMClassifier
from naive_bayes import NaiveBayesClassifier
from knn import KNNClassifier
from sklearn import metrics
import numpy as np


def read_dataset(subset):
    ''' this function reads the given subset in the data and returns lists
        of the words, labels and sentence numbers that are found in the file '''
    print('***** Reading the dataset *****')
    fname = os.path.join("lid_spaeng", f'{subset}.conll')
    words, labels, numbers = [], [], []
    with open(fname, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r"#\s*sent_enum\s*=\s*(\d+)", line)
            if match:
                sentence_number = match.group(1)
            else:
                splitted_line = line.split()
                if len(splitted_line) == 2:
                    words.append(splitted_line[0])
                    labels.append(splitted_line[1])
                    numbers.append(sentence_number)
    assert len(words) == len(labels) and len(labels) == len(
        numbers), 'Error: there should be equal number of texts, labels and sentence numbers.'
    print(f'Number of samples: {len(words)}')
    return np.array(words), np.array(labels), np.array(numbers)


def preprocess(word_list):
    ''' this function returns the preprocessed version of the word list'''
    print(f"Preprocessing the word list...")
    preprocessed_words = []
    for word in word_list:
        lowercase_word = word.lower()
        preprocessed_words.append(lowercase_word)

    print(f"Finished preprocessing for word list.")

    return preprocessed_words


def evaluate(true_labels, predicted_labels, class_labels=None):
    confusion_matrix = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    print('***** Evaluation *****')
    print(confusion_matrix)

    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)

    precision = metrics.precision_score(true_labels, predicted_labels, average='macro')
    print("Macro precision score:", precision)

    recall = metrics.recall_score(true_labels, predicted_labels, average='macro')
    print("Macro recall score:", recall)

    f1_score = metrics.f1_score(true_labels, predicted_labels, average='macro')
    print("F1-score:", f1_score)


def train_test(classifier='svm'):
    train_text, train_labels, train_num = read_dataset('train')
    test_text, test_labels, test_num = read_dataset('dev')

    train_text, train_labels = train_text[:50000], train_labels[:50000]
    test_text, test_labels = test_text[:50000], test_labels[:50000]

    train_text = preprocess(train_text)
    test_text = preprocess(test_text)

    if classifier == 'svm':
        cls = SVMClassifier()
    elif classifier == 'naive_bayes':
        cls = NaiveBayesClassifier()
    elif classifier == 'knn':
        cls = KNNClassifier()
    else:
        raise ValueError('Invalid classifier name')

    # Generate features from train and test data
    # features: character count features as a 2D numpy array, in tf-idf form
    print("Generating features...")
    train_feats = cls.tf_idf(cls.get_features(train_text))
    test_feats = cls.tf_idf(cls.get_features(test_text))

    print("Fit training data for classifier...")
    cls.fit(train_feats, train_labels)

    print("Predicting the labels")
    predicted_test_labels = cls.predict(test_feats)

    evaluate(test_labels, predicted_test_labels)
    # print(predicted_test_labels[:100], test_labels[:100])

    return cls


def main():
    # print("Running the svm classifier...")
    # train_test('svm')
    # print("Running the naive bayes classifier...")
    # train_test('naive_bayes')
    print("Running the knn classifier...")
    train_test('knn')
    # words, labels, numbers = read_dataset('train')
    # words, labels, numbers = words[:1000], labels[:1000], numbers[:1000]


    # if uncommented --> makes baseline labels and prints its accuracy
    # baseline_labels = get_baseline(words)
    # accuracy = metrics.accuracy_score(labels, baseline_labels)
    # print(accuracy)
    # evaluate(labels, baseline_labels)


if __name__ == "__main__":
    main()