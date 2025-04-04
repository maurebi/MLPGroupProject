# Machine Learning Project
This project implements a machine learning pipeline for classifying text data, specifically targeting language identification and named entity/punctuation recognition in a Spanish-English mixed dataset. It includes multiple classifiers (SVM, Naive Bayes, KNN) and ensemble methods (Voting and Stacking) built on top of a custom feature extraction framework.

Authors: Nikki van Gurp, Ilse Kerkhove, Dertje Roggeveen, Marieke Schelhaas

## Overview
The project processes a dataset in CoNLL format (e.g., train.conll, dev.conll) and performs the following tasks:

Data Reading: Extracts words, labels, and tweet information from the dataset.
Feature Extraction: Combines TF-IDF, handcrafted features (e.g., accents, capitalization), and named entity/punctuation labels.
Classification: Implements baseline (using py3langid) and advanced classifiers (SVM, Naive Bayes, KNN, Voting Ensemble, Stacking Ensemble).
Evaluation: Computes accuracy, precision, recall, and F1-score metrics.
The codebase is modular, with a base CustomClassifier class extended by specific classifier implementations.

## Installation
1. Clone the repository
```py
git clone https://github.com/maurebi/MLPGroupProject.git
```

2. Set up a virtual environment (optional)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements
```

4. Install spaCy model
```
python -m spacy download en_core_web_sm
```

5. Download NLTK data
```
python -m nltk.downloader words stopwords
```

## Usage
1. Place your dataset files (e.g., train.conll, dev.conll) in a lid_spaeng subdirectory relative to the main script.
The dataset should follow the CoNLL format with sentence numbers (# sent_enum = <number>) and word-label pairs.

2. The main script is designed to train and evaluate classifiers on the dataset.
By default, it runs the StackingEnsembleClassifier. Uncomment other classifier runs in main() to test them.
```
python main.py
```

3. The script prints dataset reading progress, feature extraction status, and evaluation metrics (confusion matrix, accuracy, precision, recall, F1-score) for the selected classifier.

## Project Structure
* custom_classifier.py: Abstract base class with feature extraction methods (TF-IDF, handcrafted features, etc.).
* baseline.py: Baseline model using py3langid for language identification.
* ne_punct_recognition.py: Named entity and punctuation recognition using spaCy.
* svm_classifier.py: SVM classifier with One-vs-One strategy.
* naive_bayes.py: Naive Bayes classifier using MultinomialNB.
* knn.py: KNN classifier with customizable neighbors and distance metric.
* ensemble_classifier.py: Voting and Stacking ensemble classifiers combining SVM, Naive Bayes, and KNN.
* main.py: Main script for dataset reading, preprocessing, training, and evaluation.
