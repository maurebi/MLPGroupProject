import py3langid as langid
import string

def get_baseline(word_list):
    ''' This function is the baseline model and returns the labels. It uses 
        the labels that langid predicts. Langid returns the language
        of which it is the highest chance the word is from. '''
    # https://arturosbn.medium.com/pre-trained-python-model-for-easy-language-identification-5630029b9cbf
    # https://pypi.org/project/py3langid/
    print("- Predicts baseline labels...")
    labels = []
    for word in word_list:
        if any(ch in string.punctuation for ch in word):
            labels.append('other')
        else:
            classification = langid.classify(word)
            if classification[0] == 'en':
                labels.append('lang1')
            elif classification[0] == 'es':
                labels.append('lang2')
            else:
                labels.append('other')
    return labels

