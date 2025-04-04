import abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import string
from ne_punct_recognition import ne_punct_recognizer
from nltk.corpus import words, stopwords

class CustomClassifier(abc.ABC):
    """Abstract base class for custom classifiers with feature extraction methods."""
    def __init__(self):
        """Initialize the CustomClassifier with word dictionaries and feature extractor."""
        self.counter = None
        self.english_words = set(words.words())
        self.spanish_stopwords = set(stopwords.words('spanish'))

    def get_features(self, text_list):
        """Extract word or n-gram count features from text.

        Parameters
        ----------
        text_list : array-like
            List or array of text strings to process.

        Returns
        -------
        sparse matrix of shape (n_samples, n_features)
            Count features as a sparse matrix.
        """
        if self.counter is None:
            self.counter = CountVectorizer(analyzer='char', ngram_range=(1,3), max_features=10000, min_df=5)
            features_array = self.counter.fit_transform(text_list)
        else:
            features_array = self.counter.transform(text_list)

        return features_array

    def tf_idf(self, text_feats):
        """Apply TF-IDF transformation to text features.

        Parameters
        ----------
        text_feats : sparse matrix of shape (n_samples, n_features)
            Count features to transform.

        Returns
        -------
        sparse matrix of shape (n_samples, n_features)
            TF-IDF transformed features.
        """
        tfidf_transformer = TfidfTransformer().fit(text_feats)
        return tfidf_transformer.transform(text_feats)

    def extract_handcrafted_features(self, word_list):
        """Extract handcrafted features from a list of words.

        Parameters
        ----------
        word_list : array-like
            List or array of words to process.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Array of handcrafted features.
        """
        print("Getting handcrafted features...")
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

        print("Finished getting handcrafted features.")

        return np.array(features)

    def get_punctuation_features(self, punct_labels):
        """Extract punctuation features from labels.

        Parameters
        ----------
        punct_labels : list
            List of labels indicating punctuation ('other') or not.

        Returns
        -------
        ndarray of shape (n_samples, 1)
            Array of binary punctuation features.
        """
        punct_features = [1 if label == 'other' else 0 for label in punct_labels]
        return np.array(punct_features).reshape(-1, 1)

    def get_named_entity_features(self, ne_labels):
        """Extract named entity features from labels.

        Parameters
        ----------
        ne_labels : list
            List of labels indicating named entities ('ne') or not.

        Returns
        -------
        ndarray of shape (n_samples, 1)
            Array of binary named entity features.
        """
        ne_features = [1 if label == 'ne' else 0 for label in ne_labels]
        return np.array(ne_features).reshape(-1, 1)

    def get_combined_features(self, text, tweet_num, tweets):
        """Combine TF-IDF, punctuation, named entity, and handcrafted features.

        Parameters
        ----------
        text : array-like
            List or array of text or words to process.
        tweet_num : array-like
            Tweet indices corresponding to the text.
        tweets : array-like
            Full tweet texts.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Combined feature array.
        """
        text_tfidf = self.tf_idf(self.get_features(text))
        text_tfidf = np.array(text_tfidf.toarray())

        ne_punct_labels = ne_punct_recognizer(text, tweet_num, tweets)
        punct_features = self.get_punctuation_features(ne_punct_labels)
        ne_features = self.get_named_entity_features(ne_punct_labels)

        handcrafted_feats = self.extract_handcrafted_features(text)

        combined_feats = np.hstack([text_tfidf, punct_features, ne_features, handcrafted_feats])
        print("Finished getting combined features.")
        return combined_feats

    @abc.abstractmethod
    def fit(self, train_features, train_labels):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass

    @abc.abstractmethod
    def predict(self, test_features):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass
