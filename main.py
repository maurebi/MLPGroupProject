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
    ''' this function reads the given subset in the data and returns lists
        of the words, labels and sentence numbers that are found in the file '''
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
    ''' this function returns the preprocessed version of the word list'''
    print(f"- Preprocessing the word list...")
    preprocessed_words = []
    for word in word_list:
        lowercase_word = word.lower()
        preprocessed_words.append(lowercase_word)

    print(f"- Finished preprocessing for word list. ")


    return preprocessed_words


def evaluate(true_labels, predicted_labels, class_labels=None):
    '''This function evaluates the models based on 
    accuracy, recall, precision and F1-scores'''
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

    if classifier == 'svm':
        cls = SVMClassifier()
    elif classifier == 'naive_bayes':
        cls = NaiveBayesClassifier()
    elif classifier == 'knn':
        cls = KNNClassifier()
    elif classifier == 'voting':
        cls = VotingEnsembleClassifier()
    # elif classifier == 'stacking':
    #     cls = StackingEnsembleClassifier()
    else:
        raise ValueError('Invalid classifier name')

    if classifier == 'voting' or classifier == 'stacking':
        # Processing `print-statements are in the function itself`
        cls.fit(train_labels, train_text, train_num, train_tweets)
        print("- Predicting the labels")
        predicted_test_labels = cls.predict(test_text, test_num, test_tweets)
    else:
        # Generate combined features using get_combined_features for both train and test data
        train_feats = cls.tf_idf(cls.get_features(train_text))
        test_feats = cls.tf_idf(cls.get_features(test_text))
        print("- Fit training data for classifier...")
        cls.fit(train_feats, train_labels)
        print("- Predicting the labels")
        predicted_test_labels = cls.predict(test_feats)

    evaluate(test_labels, predicted_test_labels)

    return cls


def main():
    train_text, train_labels, train_num, train_tweets = read_dataset('train')
    test_text, test_labels, test_num, test_tweets = read_dataset('dev')

    # train_text, train_labels, train_num, train_tweets = train_text[:1000], train_labels[:1000], train_num[:1000], train_tweets[:1000]
    # test_text, test_labels, test_num, test_tweets = test_text[:200], test_labels[:200], test_num[:200], test_tweets[:200]

    # train_text, train_labels, train_num, train_tweets = train_text[:10000], train_labels[:10000], train_num[:10000], train_tweets[:10000]
    # test_text, test_labels, test_num, test_tweets = test_text[:2000], test_labels[:2000], test_num[:2000], test_tweets[:2000]
    
    # train_text = preprocess(train_text)
    # test_text = preprocess(test_text)
    
    print("*** Running the baseline...")
    baseline_labels = get_baseline(train_text)  
    accuracy = metrics.accuracy_score(train_labels, baseline_labels)
    print(accuracy)
    evaluate(train_labels, baseline_labels)
    
    
    # print("*** Running the svm classifier...")
    # train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, 'svm')

    # print("*** Running the naive bayes classifier...")
    # train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, 'naive_bayes')

    # print("*** Running the knn classifier...")
    # train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, 'knn')

    # print("*** Running the voting ensemble classifier...")
    # train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, classifier='voting')

    # print("*** Running the stacking ensemble classifier...")
    # train_test(train_text, test_text, train_labels, train_num, train_tweets, test_num, test_tweets, test_labels, 'stacking')


if __name__ == "__main__":
    main()