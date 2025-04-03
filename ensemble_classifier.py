from custom_classifier import CustomClassifier
from svm_classifier import SVMClassifier
from naive_bayes import NaiveBayesClassifier
from knn import KNNClassifier
from ne_punct_recognition import ne_punct_recognizer
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np



class VotingEnsembleClassifier(CustomClassifier):
    def __init__(self):
        super().__init__()
        self.svm = SVMClassifier()
        self.nb = NaiveBayesClassifier()
        self.knn = KNNClassifier()

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
    
    def get_combined_features(self, train_text, tweet_num, tweets):
        text_tfidf = self.tf_idf(self.get_features(train_text))
        text_tfidf = np.array(text_tfidf.toarray())
        ne_punct_labels = ne_punct_recognizer(train_text, tweet_num, tweets)     
        punct_features = self.get_punctuation_features(ne_punct_labels)
        ne_features = self.get_named_entity_features(ne_punct_labels)
    
        combined_feats = np.hstack([text_tfidf, punct_features, ne_features])

        return combined_feats

    def get_punctuation_features(self, punct_labels):   
        punct_features = [1 if label == 'other' else 0 for label in punct_labels]  

        return np.array(punct_features).reshape(-1, 1)  
    
    def get_named_entity_features(self, ne_labels):
        ne_features = [1 if label == 'ne' else 0 for label in ne_labels]
        
        return np.array(ne_features).reshape(-1, 1)



class StackingEnsembleClassifier(CustomClassifier):
    def __init__(self):
        super().__init__()
        self.svm = SVMClassifier()
        self.nb = NaiveBayesClassifier()
        self.knn = KNNClassifier()

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
    
    def get_combined_features(self, train_text, tweet_num, tweets):
        text_tfidf = self.tf_idf(self.get_features(train_text))
        text_tfidf = np.array(text_tfidf.toarray())
        ne_punct_labels = ne_punct_recognizer(train_text, tweet_num, tweets)     
        punct_features = self.get_punctuation_features(ne_punct_labels)
        ne_features = self.get_named_entity_features(ne_punct_labels)
    
        combined_feats = np.hstack([text_tfidf, punct_features, ne_features])

        return combined_feats

    def get_punctuation_features(self, punct_labels):   
        punct_features = [1 if label == 'other' else 0 for label in punct_labels]  

        return np.array(punct_features).reshape(-1, 1)  
    
    def get_named_entity_features(self, ne_labels):
        ne_features = [1 if label == 'ne' else 0 for label in ne_labels]
        
        return np.array(ne_features).reshape(-1, 1)
