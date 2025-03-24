import numpy as np
import scipy
from CustomClassifier import CustomClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
Implement a KNN classifier with required functions:

fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class KNNClassifier(CustomClassifier):

    def __init__(self, k=5, distance_metric='cosine'):
        """ """
        super().__init__()

        self.k = k
        self.train_feats = None
        self.train_labels = None
        self.is_trained = False
        self.distance_metric = distance_metric
        self.knn = KNeighborsClassifier(n_neighbors=self.k, metric=self.distance_metric)

    def fit(self, train_feats, train_labels):
        """ Fit training data for classifier """
        
        # if hasattr(train_feats, "toarray"):
        #     train_feats = train_feats.toarray()

        self.train_feats = train_feats
        self.train_labels = np.array(train_labels)
        
        self.knn.fit(self.train_feats, self.train_labels)

        self.is_trained = True
        return self
        

    def predict(self, test_feats):
        """ Predict classes with provided test features """

        assert self.is_trained, 'Model must be trained before predicting'

        # if hasattr(test_feats, "toarray"):
        #     test_feats = test_feats.toarray()

        return self.knn.predict(test_feats)