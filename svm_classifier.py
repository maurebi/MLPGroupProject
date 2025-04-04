from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from custom_classifier import CustomClassifier


class SVMClassifier(CustomClassifier):
    """A Support Vector Machine classifier extending CustomClassifier using OneVsOne strategy."""
    def __init__(self, kernel='linear'):
        """Initialize the SVM classifier.

        Parameters
        ----------
        kernel : str, optional
            Kernel type to be used in the SVM algorithm (default is 'linear').
        """
        super().__init__()
        self.counter = None
        self.classifier = OneVsOneClassifier(svm.LinearSVC(class_weight='balanced', dual=False, max_iter=5000))
        # self.classifier = OneVsOneClassifier(svm.SVC(kernel='rbf', class_weight='balanced', probability=True)) # Might be better but takes very long)

    def fit(self, train_features, train_labels):
        """Fit the SVM model to the training data.

        Parameters
        ----------
        train_features : array-like of shape (n_samples, n_features)
            Training data features.
        train_labels : array-like of shape (n_samples,)
            Training data labels.

        Returns
        -------
        self : SVMClassifier
            Returns the instance itself after fitting.
        """
        print("Fitting svm model...")
        self.classifier.fit(train_features, train_labels)
        print("Finished fitting svm model.")
        return self

    def predict(self, test_features):
        """Predict labels for the test data using the trained model.

        Parameters
        ----------
        test_features : array-like of shape (n_samples, n_features)
            Test data features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels for the test data.
        """
        print("Predicting svm model...")
        return self.classifier.predict(test_features)
