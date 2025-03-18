# Machine Learning Project
# Nikki van Gurp, Ilse Kerkhove, Dertje Roggeveen & Marieke Schelhaas
import os
import re


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


def main():
    words, labels, numbers = read_dataset('train')
    # print(words)
    # print(labels)
    # print(numbers)
    


if __name__ == "__main__":
    main()