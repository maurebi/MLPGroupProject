import abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CustomClassifier(abc.ABC):
    def __init__(self):
        self.counter = None

    
    def get_features(self, text_list):
        """ Return word (or ngram) count features for each text as a 2D numpy array """
        if self.counter is None:
            self.counter = CountVectorizer(analyzer='char', ngram_range=(1,3))
            features_array = self.counter.fit_transform(text_list)
        else:
            features_array = self.counter.transform(text_list)

        return features_array


    def tf_idf(self, text_feats):
        tfidf_transformer = TfidfTransformer().fit(text_feats)
        return tfidf_transformer.transform(text_feats)
    

    @abc.abstractmethod
    def fit(self, train_features, train_labels):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass

    @abc.abstractmethod
    def predict(self, test_features):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass
