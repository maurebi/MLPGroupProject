from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from custom_classifier import CustomClassifier



class SVMClassifier(CustomClassifier):
    def __init__(self, kernel='linear'):
        super().__init__()
        self.counter = None
        self.classifier = OneVsOneClassifier(svm.LinearSVC(class_weight='balanced', dual=False))

    def fit(self, train_features, train_labels):
        self.classifier.fit(train_features, train_labels)

    def predict(self, test_features):
        return self.classifier.predict(test_features)
