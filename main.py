# Machine Learning Project
# Nikki van Gurp, Ilse Kerkhove, Dertje Roggeveen & Marieke Schelhaas
import os
import re
import pandas as pd
import numpy as np
import spacy
import sklearn
import time
from scipy.sparse import csr_matrix, hstack
from baseline import get_baseline
from svm_classifier import SVMClassifier
from naive_bayes import NaiveBayesClassifier
from knn import KNNClassifier
from sklearn import metrics
from ensemble_classifier import VotingEnsembleClassifier, StackingEnsembleClassifier

def read_dataset(subset):
    """Read the given subset from the dataset and extract words, labels, and sentence numbers.

    Parameters
    ----------
    subset : str
        The name of the dataset subset to read (e.g., 'train', 'dev').

    Returns
    -------
    ndarray
        Array of words extracted from the dataset.
    ndarray
        Array of labels corresponding to the words.
    ndarray
        Array of sentence numbers corresponding to the words.
    ndarray
        Array of full tweets constructed from the words.
    """
    print('***** Reading the dataset *****')
    fname = os.path.join("lid_spaeng", f'{subset}.conll')
    words, labels, numbers, tweets = [], [], [], []
    currentTweet = None
    tweet = ""
    
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
                    if sentence_number != currentTweet:
                        currentTweet = sentence_number
                        tweets.append(tweet)            
                        tweet = ""
                    else:
                        tweet += " " + splitted_line[0]
    assert len(words) == len(labels) and len(labels) == len(
        numbers), 'Error: there should be equal number of texts, labels and sentence numbers.'

    print(f'Number of samples: {len(words)}')
    return np.array(words), np.array(labels), np.array(numbers), np.array(tweets)



def preprocess(word_list):
    """Preprocess a list of words by converting them to lowercase.

    Parameters
    ----------
    word_list : list or ndarray
        List or array of words to preprocess.

    Returns
    -------
    list
        List of preprocessed (lowercased) words.
    """
    print(f"- Preprocessing the word list...")
    preprocessed_words = []
    for word in word_list:
        lowercase_word = word.lower()
        preprocessed_words.append(lowercase_word)

    print(f"- Finished preprocessing for word list. ")


    return preprocessed_words


def evaluate(true_labels, predicted_labels, class_labels=None):
    """Evaluate model performance using accuracy, precision, recall, and F1-score metrics.

    Parameters
    ----------
    true_labels : array-like
        Ground truth labels for the test data.
    predicted_labels : array-like
        Predicted labels from the model.
    class_labels : list, optional
        List of class labels. If None, inferred from true_labels and predicted_labels.

    Returns
    -------
    None
        Prints evaluation metrics including confusion matrix, accuracy, precision, recall, and F1-score.
    """
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


def train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, classifier='svm'):    
    """Train a classifier and evaluate its performance on test data.

    Parameters
    ----------
    train_text : array-like
        Training data words or text features.
    test_text : array-like
        Test data words or text features.
    train_labels : array-like
        Labels for the training data.
    train_num : array-like
        Sentence numbers for the training data.
    train_tweets : array-like
        Full tweets for the training data.
    test_num : array-like
        Sentence numbers for the test data.
    test_tweets : array-like
        Full tweets for the test data.
    test_labels : array-like
        Labels for the test data.
    classifier : str, optional
        Type of classifier to use ('svm', 'naive_bayes', 'knn', 'voting', 'stacking'). Default is 'svm'.

    Returns
    -------
    object
        Trained classifier instance.
    """
    if classifier == 'svm':
        cls = SVMClassifier()
    elif classifier == 'naive_bayes':
        cls = NaiveBayesClassifier()
    elif classifier == 'knn':
        cls = KNNClassifier()
    elif classifier == 'voting':
        cls = VotingEnsembleClassifier()
    elif classifier == 'stacking':
        cls = StackingEnsembleClassifier()
    else:
        raise ValueError('Invalid classifier name')

    # Train and predict
    if classifier in ['voting', 'stacking']:
        cls.fit(train_labels, train_text, train_num, train_tweets)

        predicted_test_labels = cls.predict(test_text, test_num, test_tweets)
    else:
        train_feats = cls.get_combined_features(train_text, train_num, train_tweets)
        test_feats = cls.get_combined_features(test_text, test_num, test_tweets)

        cls.fit(train_feats, train_labels)
        predicted_test_labels = cls.predict(test_feats)

    evaluate(test_labels, predicted_test_labels)

    return cls




def main():
    train_text, train_labels, train_num, train_tweets = read_dataset('train')
    test_text, test_labels, test_num, test_tweets = read_dataset('dev')

    # train_text, train_labels, train_num, train_tweets = train_text[:1000], train_labels[:1000], train_num[:1000], train_tweets[:1000]
    # test_text, test_labels, test_num, test_tweets = test_text[:200], test_labels[:200], test_num[:200], test_tweets[:200]

    # train_text, train_labels, train_num, train_tweets = train_text[:50000], train_labels[:50000], train_num[:50000], train_tweets[:50000]
    # test_text, test_labels, test_num, test_tweets = test_text[:2000], test_labels[:2000], test_num[:2000], test_tweets[:2000]
    
    # train_text = preprocess(train_text)
    # test_text = preprocess(test_text)
    
    # TODO does not currently work --> FIX
    # print("*** Running the baseline...")
    # baseline_labels = get_baseline(train_test)  
    # accuracy = metrics.accuracy_score(train_labels, baseline_labels)
    # print(accuracy)
    # evaluate(train_labels, baseline_labels)
    
    
    # print("*** Running the svm classifier...")
    # train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, 'svm')

    # print("*** Running the naive bayes classifier...")
    # train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, 'naive_bayes')

    # print("*** Running the knn classifier...")
    # train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, 'knn')

    # print("*** Running the voting ensemble classifier...")
    # train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, classifier='voting')

    print("*** Running the stacking ensemble classifier...")
    train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, 'stacking')


if __name__ == "__main__":
    main()