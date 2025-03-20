import numpy as np
from CustomClassifier import CustomClassifier
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesClassifier(CustomClassifier):
    """
    Custom Naive Bayes classifier using scikit-learn's MultinomialNB.
    """

    def __init__(self, alpha=1.0):
        """ Initialize the Naive Bayes classifier with smoothing
        parameter alpha. """
        super().__init__()
        self.alpha = alpha
        self.model = MultinomialNB(class_prior=self.prior_probabilities, alpha=self.alpha)
        self.prior_probabilities = None
        self.is_trained = False

    def fit(self, X_train, y_train):
        """
        Train the Naive Bayes classifier.

        Fits the Naive Bayes model to the provided training data and
        computes the prior probabilities of each class.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Feature matrix for training data.
        y_train : array-like of shape (n_samples,)
            Corresponding labels for the training data.

        Returns
        -------
        CustomNaiveBayes
            The trained instance of `CustomNaiveBayes`.

        Notes
        -----
        This method calculates the class prior probabilities based on the
        distribution of `y_train` and stores them in `prior_probabilities`.
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Compute class priors
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        self.prior_probabilities = class_counts / np.sum(class_counts)

        return self

    def predict(self, X_test):
        """
        Predict class labels for the given test data.

        Uses the trained Naive Bayes model to predict class labels for
        input feature vectors.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Feature matrix for test data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels for each sample.

        Raises
        ------
        ValueError
            If the model has not been trained before prediction.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained first.")

        return self.model.predict(X_test)
