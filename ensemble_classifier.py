from custom_classifier import CustomClassifier
from svm_classifier import SVMClassifier
from naive_bayes import NaiveBayesClassifier
from knn import KNNClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

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

    def fit(self, train_features, train_labels):
        # fit all individual classifiers
        self.svm.fit(train_features, train_labels)
        self.nb.fit(train_features, train_labels)
        self.knn.fit(train_features, train_labels)

        self.ensemble.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.ensemble.predict(test_features)


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

    def fit(self, train_features, train_labels):
        # fit all individual classifiers
        self.svm.fit(train_features, train_labels)
        self.nb.fit(train_features, train_labels)
        self.knn.fit(train_features, train_labels)

        self.ensemble.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.ensemble.predict(test_features)
