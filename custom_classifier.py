import abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import string
from ne_punct_recognition import ne_punct_recognizer
from nltk.corpus import words, stopwords

class CustomClassifier(abc.ABC):
    def __init__(self):
        self.counter = None
        self.english_words = set(words.words())
        self.spanish_stopwords = set(stopwords.words('spanish'))

    def get_features(self, text_list):
        """ Return word (or ngram) count features for each text as a 2D numpy array """
        if self.counter is None:
            self.counter = CountVectorizer(analyzer='char', ngram_range=(3,3), max_features=10000, min_df=5)
            features_array = self.counter.fit_transform(text_list)
        else:
            features_array = self.counter.transform(text_list)

        return features_array

    def tf_idf(self, text_feats):
        tfidf_transformer = TfidfTransformer().fit(text_feats)
        return tfidf_transformer.transform(text_feats)

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

    def get_punctuation_features(self, punct_labels):
        punct_features = [1 if label == 'other' else 0 for label in punct_labels]
        return np.array(punct_features).reshape(-1, 1)

    def get_named_entity_features(self, ne_labels):
        ne_features = [1 if label == 'ne' else 0 for label in ne_labels]
        return np.array(ne_features).reshape(-1, 1)

    def get_combined_features(self, text, tweet_num, tweets):
        text_tfidf = self.tf_idf(self.get_features(text))
        text_tfidf = np.array(text_tfidf.toarray())

        ne_punct_labels = ne_punct_recognizer(text, tweet_num, tweets)
        punct_features = self.get_punctuation_features(ne_punct_labels)
        ne_features = self.get_named_entity_features(ne_punct_labels)

        handcrafted_feats = self.extract_handcrafted_features(text)

        combined_feats = np.hstack([text_tfidf, punct_features, ne_features, handcrafted_feats])
        return combined_feats

    @abc.abstractmethod
    def fit(self, train_features, train_labels):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass

    @abc.abstractmethod
    def predict(self, test_features):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass
