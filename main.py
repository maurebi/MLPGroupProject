# Machine Learning Project
# Nikki van Gurp, Ilse Kerkhove, Dertje Roggeveen & Marieke Schelhaas
import os
import re
from baseline import *
from sklearn import metrics


def read_dataset(subset):
    ''' this function reads the given subset in the data and returns lists
        of the words, labels and sentence numbers that are found in the file '''
    print('***** Reading the dataset *****')
    fname = os.path.join("lid_spaeng", f'{subset}.conll')
    words, labels, numbers = [], [], []
    with open(fname, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r"#\s*sent_enum\s*=\s*(\d+)", line)
            if match:
                sentence_number = match.group(1)
            else:
                splitted_line = line.split()
                if len(splitted_line) == 2:
                    words.append(splitted_line[0])
                    labels.append(splitted_line[1])
                    numbers.append(sentence_number)
    assert len(words) == len(labels) and len(labels) == len(numbers), 'Error: there should be equal number of texts, labels and sentence numbers.'
    print(f'Number of samples: {len(words)}')
    return words, labels, numbers

def preprocess(word_list):
    ''' this function returns the preprocessed version of the word list'''
    preprocessed_words = []
    for word in word_list:
        lowercase_word = word.lower()
        preprocessed_words.append(lowercase_word)

    return preprocessed_words


def read_dataset(subset, split):
    pass


def preprocess_dataset(text_list):
    pass


def main():
    words, labels, numbers = read_dataset('train')
    preprocessed = preprocess(words)
    # print(preprocessed)
    # print(words)
    # print(labels)
    # print(numbers)

    # if uncommented --> makes baseline labels and prints its accuracy
    # baseline_labels = get_baseline(words) # duurt lang
    # accuracy = metrics.accuracy_score(labels, baseline_labels)
    # print(accuracy)    

if __name__ == "__main__":
    main()