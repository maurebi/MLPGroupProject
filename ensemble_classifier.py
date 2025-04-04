from custom_classifier import CustomClassifier
from svm_classifier import SVMClassifier
from naive_bayes import NaiveBayesClassifier
from knn import KNNClassifier
from ne_punct_recognition import ne_punct_recognizer
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
from nltk.corpus import words, stopwords
import string

#TODO update requirements.txt


class VotingEnsembleClassifier(CustomClassifier):
    def __init__(self):
        super().__init__()
        self.svm = SVMClassifier()
        self.nb = NaiveBayesClassifier()
        self.knn = KNNClassifier()
        self.english_words = set(words.words())
        self.spanish_stopwords = set(stopwords.words('spanish'))

        self.ensemble = VotingClassifier(
            estimators=[
                ('svm', self.svm.classifier),
                ('naive_bayes', self.nb.model),
                ('knn', self.knn.knn)
            ],
            voting='hard'
        )

    def fit(self, train_labels, train_text, tweet_num, tweets):
        print("- Generating features...")
        combined_train_features = self.get_combined_features(train_text, tweet_num, tweets)
        
        # fit all individual classifiers
        print("- Fitting individual classifiers...")
        self.svm.fit(combined_train_features, train_labels)
        self.nb.fit(combined_train_features, train_labels)
        self.knn.fit(combined_train_features, train_labels)

        # fit the ensemble
        print("- Fitting ensemble classifier...")
        self.ensemble.fit(combined_train_features, train_labels)

    def predict(self, test_text, tweet_num, tweets):
        combined_test_features = self.get_combined_features(test_text, tweet_num, tweets)

        return self.ensemble.predict(combined_test_features)

    def extract_handcrafted_features(self, word_list):
        print("Extracting handcrafted features...")
        features = []
        words_left = len(word_list)

        for word in word_list:
            print(f"Number of words left is {words_left}")
            word_lower = word.lower()
            has_accent = any(c in "áéíóúñ" for c in word_lower)
            is_ascii = all(ord(c) < 128 for c in word_lower)
            is_capitalized = word[0].isupper() if word else False
            only_special_chars = all(c in string.punctuation for c in word)
            contains_numbers = any(c.isdigit() for c in word)
            avg_unicode = np.mean([ord(c) for c in word_lower]) if word else 0
            
            in_english_dict = int(word_lower in self.english_words)
            in_spanish_dict = int(word_lower in self.spanish_stopwords)
            in_both_dicts = int(in_english_dict and in_spanish_dict)
            in_neither = int(not in_english_dict and not in_spanish_dict)

            features.append([
                int(has_accent),
                int(is_ascii),
                int(is_capitalized),
                int(only_special_chars),
                int(contains_numbers),
                avg_unicode,
                in_english_dict,
                in_spanish_dict,
                in_both_dicts,
                in_neither,
            ])
        
            words_left = words_left - 1
        
        print("Finished getting handcrafted features.")
        
        return np.array(features)
    
    def get_combined_features(self, train_text, tweet_num, tweets):
        print("Getting combined features...")
        text_tfidf = self.tf_idf(self.get_features(train_text))
        text_tfidf = np.array(text_tfidf.toarray())
        
        ne_punct_labels = ne_punct_recognizer(train_text, tweet_num, tweets)     
        punct_features = self.get_punctuation_features(ne_punct_labels)
        ne_features = self.get_named_entity_features(ne_punct_labels)
        
        handcrafted_feats = self.extract_handcrafted_features(train_text)

        combined_feats = np.hstack([text_tfidf, punct_features, ne_features, handcrafted_feats])
        print("Finished getting combined features...")

        return combined_feats


    def get_punctuation_features(self, punct_labels):
        print("Getting punctuation features...")
        punct_features = [1 if label == 'other' else 0 for label in punct_labels]
        print("Finished getting punctuation features.") 

        return np.array(punct_features).reshape(-1, 1)  
    
    def get_named_entity_features(self, ne_labels):
        print("Getting named entity features...")
        ne_features = [1 if label == 'ne' else 0 for label in ne_labels]
        print("Finished getting named entity features.")
        
        return np.array(ne_features).reshape(-1, 1)



class StackingEnsembleClassifier(CustomClassifier):
    def __init__(self):
        super().__init__()
        self.svm = SVMClassifier()
        self.nb = NaiveBayesClassifier()
        self.knn = KNNClassifier()
        self.english_words = set(words.words())
        self.spanish_stopwords = set(stopwords.words('spanish'))

        self.ensemble = StackingClassifier(
            estimators=[
                ('svm', self.svm.classifier),
                ('naive_bayes', self.nb.model),
                ('knn', self.knn.knn)
            ],
            final_estimator=LogisticRegression(),
            passthrough=True
        )

    def fit(self, train_labels, train_text, tweet_num, tweets):
        print("- Generating features...")
        combined_train_features = self.get_combined_features(train_text, tweet_num, tweets)
        
        # fit all individual classifiers
        print("- Fitting individual classifiers...")
        self.svm.fit(combined_train_features, train_labels)
        self.nb.fit(combined_train_features, train_labels)
        self.knn.fit(combined_train_features, train_labels)

        # fit the ensemble
        print("- Fitting ensemble classifier...")
        self.ensemble.fit(combined_train_features, train_labels)

    def predict(self, test_text, tweet_num, tweets):
        combined_test_features = self.get_combined_features(test_text, tweet_num, tweets)

        return self.ensemble.predict(combined_test_features)

    def extract_handcrafted_features(self, word_list):
        features = []

        for word in word_list:
            word_lower = word.lower()
            has_accent = any(c in "áéíóúñ" for c in word_lower)
            is_ascii = all(ord(c) < 128 for c in word_lower)
            is_capitalized = word[0].isupper() if word else False
            only_special_chars = all(c in string.punctuation for c in word)
            contains_numbers = any(c.isdigit() for c in word)
            avg_unicode = np.mean([ord(c) for c in word_lower]) if word else 0

            in_english_dict = int(word_lower in self.english_words)
            in_spanish_dict = int(word_lower in self.spanish_stopwords)
            in_both_dicts = int(in_english_dict and in_spanish_dict)
            in_neither = int(not in_english_dict and not in_spanish_dict)

            features.append([
                int(has_accent),
                int(is_ascii),
                int(is_capitalized),
                int(only_special_chars),
                int(contains_numbers),
                avg_unicode,
                in_english_dict,
                in_spanish_dict,
                in_both_dicts,
                in_neither,
            ])
        
        return np.array(features)

    def get_combined_features(self, train_text, tweet_num, tweets):
        text_tfidf = self.tf_idf(self.get_features(train_text))
        text_tfidf = np.array(text_tfidf.toarray())
        
        ne_punct_labels = ne_punct_recognizer(train_text, tweet_num, tweets)     
        punct_features = self.get_punctuation_features(ne_punct_labels)
        ne_features = self.get_named_entity_features(ne_punct_labels)
        
        handcrafted_feats = self.extract_handcrafted_features(train_text)

        combined_feats = np.hstack([text_tfidf, punct_features, ne_features, handcrafted_feats])

        return combined_feats


    def get_punctuation_features(self, punct_labels):   
        punct_features = [1 if label == 'other' else 0 for label in punct_labels]  

        return np.array(punct_features).reshape(-1, 1)  
    
    def get_named_entity_features(self, ne_labels):
        ne_features = [1 if label == 'ne' else 0 for label in ne_labels]
        
        return np.array(ne_features).reshape(-1, 1)
