from custom_classifier import CustomClassifier
from svm_classifier import SVMClassifier
from naive_bayes import NaiveBayesClassifier
from knn import KNNClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB

#TODO update requirements.txt


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
        combined_train_features = self.get_combined_features(train_text, tweet_num, tweets)

        # Fit individual classifiers
        self.svm.fit(combined_train_features, train_labels)
        self.nb.fit(combined_train_features, train_labels)
        self.knn.fit(combined_train_features, train_labels)

        # Fit ensemble
        self.ensemble.fit(combined_train_features, train_labels)

    def predict(self, test_text, tweet_num, tweets):
        combined_test_features = self.get_combined_features(test_text, tweet_num, tweets)
        return self.ensemble.predict(combined_test_features)

class StackingEnsembleClassifier(CustomClassifier):
    def __init__(self):
        super().__init__()
        self.svm = SVMClassifier()
        self.nb = NaiveBayesClassifier()
        self.knn = KNNClassifier()

        # self.ensemble = StackingClassifier(
        #     estimators=[
        #         ('svm', self.svm.classifier),
        #         ('naive_bayes', self.nb.model),
        #         ('knn', self.knn.knn)
        #     ],
        #     final_estimator=LogisticRegression(),
        #     passthrough=True
        # )
        self.ensemble = StackingClassifier(
            estimators=[
                ('svm', self.svm.classifier),
                ('naive_bayes', self.nb.model),
                ('knn', self.knn.knn)
            ],
            final_estimator=RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=5, 
                class_weight='balanced', 
                random_state=42
            ),
            cv=5,  # Use 5-fold cross-validation for out-of-fold predictions
            passthrough=True,  # Pass original features to the meta-classifier
            n_jobs=-1  # Parallelize training
        )

    def fit(self, train_labels, train_text, tweet_num, tweets):
        combined_train_features = self.get_combined_features(train_text, tweet_num, tweets)

        self.svm.fit(combined_train_features, train_labels)
        self.nb.fit(combined_train_features, train_labels)
        self.knn.fit(combined_train_features, train_labels)

        self.ensemble.fit(combined_train_features, train_labels)

    def predict(self, test_text, tweet_num, tweets):
        combined_test_features = self.get_combined_features(test_text, tweet_num, tweets)
        return self.ensemble.predict(combined_test_features)
