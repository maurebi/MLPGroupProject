# Machine Learning Project
# Nikki van Gurp, Ilse Kerkhove, Dertje Roggeveen & Marieke Schelhaas
import os
import re
import pandas as pd
import numpy as np
import spacy
import sklearn
import time
from collections import Counter
from baseline import *
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


def train_test(train_text, test_text, train_labels, test_labels, classifier='svm'):    

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

    # Generate features from train and test data
    # features: character count features as a 2D numpy array, in tf-idf form
    print("- Generating features...")
    train_feats = cls.tf_idf(cls.get_features(train_text))
    test_feats = cls.tf_idf(cls.get_features(test_text))
    
    print("- Fit training data for classifier...")
    cls.fit(train_feats, train_labels)

    print("- Predicting the labels")
    predicted_test_labels = cls.predict(test_feats)
    print(type(predicted_test_labels))
    print(predicted_test_labels[:100])

    evaluate(test_labels, predicted_test_labels)
    # print(predicted_test_labels[:100], test_labels[:100])

    return cls


def ne_punct_recognizer(words, tweet_num, tweets):
    print(f"- Processing Named Entities and Punctuation...")
    labels = []
    currentTweet = None
    nlpEN = spacy.load('en_core_web_sm') 

    for i in range(len(words)): 
        # Keep in mind what tweet we are
        if tweet_num[i] != currentTweet:
            currentTweet = str(tweets[int(tweet_num[i])])
            doc = nlpEN(currentTweet)
        
        if any(ch in string.punctuation for ch in words[i]):
            labels.append('other')
        elif len(doc.ents) != 0:
            matched = False
            for ent in doc.ents:
                if words[i] in ent.text.split():
                    labels.append('ne')
                    matched = True
                    break
            if not matched:
                labels.append('none')
        else:
            labels.append('none')
    print(f"- Finished processing Named Entities and Punctuation...") 


    return labels


def main():
    train_text, train_labels, train_num, train_tweets = read_dataset('train')
    test_text, test_labels, test_num, test_tweets = read_dataset('dev')

    train_text = preprocess(train_text)
    test_text = preprocess(test_text)
    
    # # if uncommented --> makes baseline labels and prints its accuracy
    # baseline_labels = get_baseline(Train_words)  
    # # accuracy = metrics.accuracy_score(Train_labels, baseline_labels)
    # # print(accuracy)
    # evaluate(Train_labels, baseline_labels)
    
    output = ne_punct_recognizer(test_text[:1000], test_num[:1000], test_tweets)
    print(Counter(output))
    
    
    
    # print("Running the svm classifier...")
    # train_test(train_text, test_text, train_labels, test_labels, 'svm')
    # print("Running the naive bayes classifier...")
    # train_test(train_text, test_text, train_labels, test_labels, 'naive_bayes')
    # print("Running the knn classifier...")
    # train_test(train_text, test_text, train_labels, test_labels, 'knn')

    # print("Running the voting ensemble classifier...")
    # train_test(train_text, test_text, train_labels, test_labels, 'voting')

    # print("Running the stacking ensemble classifier...")
    # train_test(train_text, test_text, train_labels, test_labels, 'stacking')


    


    



if __name__ == "__main__":
    main()