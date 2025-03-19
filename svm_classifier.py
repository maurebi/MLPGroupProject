from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier


class SVMClassifier:
    def __init__(self):
        super().__init__()

        self.classifier = OneVsRestClassifier(svm.SVC())

    def fit(self, train_features, train_labels):
        self.classifier.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.classifier.predict(test_features)
