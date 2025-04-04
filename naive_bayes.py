import numpy as np
from custom_classifier import CustomClassifier
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesClassifier(CustomClassifier):
    """A Naive Bayes classifier extending CustomClassifier using MultinomialNB."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.prior_probabilities = None
        self.model = MultinomialNB(class_prior=self.prior_probabilities, alpha=self.alpha)
        self.is_trained = False

    def fit(self, X_train, y_train):
        """Fit the Naive Bayes model to the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data features.
        y_train : array-like of shape (n_samples,)
            Training data labels.

        Returns
        -------
        self : NaiveBayesClassifier
            Returns the instance itself after fitting.
        """
        print("Fitting nb model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Compute class priors
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        self.prior_probabilities = class_counts / np.sum(class_counts)
        print("Finished fitting nb model.")

        return self

    def predict(self, X_test):
        """Predict labels for the test data using the trained model.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels for the test data.
        """
        print("Predicting nb model...")
        if not self.is_trained:
            raise ValueError("The model must be trained first.")
        print("Finished predicting nb model.")

        return self.model.predict(X_test)
