import numpy as np
from custom_classifier import CustomClassifier
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier(CustomClassifier):
    """A K-Nearest Neighbors classifier extending CustomClassifier."""

    def __init__(self, k=5, distance_metric='cosine'):
        """Initialize the KNN classifier.

        Parameters
        ----------
        k : int, optional
            Number of neighbors to use for classification (default is 5).
        distance_metric : str, optional
            Distance metric to use for the KNN algorithm (default is 'cosine').
        """
        super().__init__()

        self.k = k
        self.train_feats = None
        self.train_labels = None
        self.is_trained = False
        self.distance_metric = distance_metric
        self.knn = KNeighborsClassifier(n_neighbors=self.k, metric=self.distance_metric)

    def fit(self, train_feats, train_labels):
        """Fit the KNN model to the training data.

        Parameters
        ----------
        train_feats : array-like of shape (n_samples, n_features)
            Training data features.
        train_labels : array-like of shape (n_samples,)
            Training data labels.

        Returns
        -------
        self : KNNClassifier
            Returns the instance itself after fitting.
        """
        print("Fitting knn model...")

        self.train_feats = train_feats
        self.train_labels = np.array(train_labels)

        self.knn.fit(self.train_feats, self.train_labels)

        self.is_trained = True
        print("KNN model fitted.")
        return self

    def predict(self, test_feats):
        """Predict labels for the test data using the trained model.

        Parameters
        ----------
        test_feats : array-like of shape (n_samples, n_features)
            Test data features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels for the test data.
        """
        print("Predicting knn model...")
        assert self.is_trained, 'Model must be trained before predicting'
        print("Finished predicting knn model.")

        return self.knn.predict(test_feats)
