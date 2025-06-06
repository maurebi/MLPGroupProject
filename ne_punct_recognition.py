import spacy
import string


nlpEN = spacy.load('en_core_web_sm')


def ne_punct_recognizer(words, tweet_num, tweets):
    """Recognize named entities and punctuation in a list of words from tweets.

    Parameters
    ----------
    words : array-like
        List or array of words to process.
    tweet_num : array-like
        Array of tweet indices corresponding to each word.
    tweets : array-like
        Array of full tweet texts.

    Returns
    -------
    list
        List of labels ('other' for punctuation, 'ne' for named entities,
        'none' otherwise).
    """
    print(f"- Processing Named Entities and Punctuation...")
    labels = []
    punctuation_set = set(string.punctuation)

    # Preprocess all tweets only once
    tweet_entities = {}
    for idx, tweet in enumerate(tweets):
        doc = nlpEN(str(tweet))
        entity_words = set()
        for ent in doc.ents:
            for token in ent.text.split():
                entity_words.add(token)
        tweet_entities[idx] = entity_words

    # Now fast lookup per word
    for i in range(len(words)):
        tweet_idx = int(tweet_num[i])
        word = words[i]
        if word in punctuation_set:
            labels.append('other')
        elif word in tweet_entities.get(tweet_idx, set()):
            labels.append('ne')
        else:
            labels.append('none')

    print(f"- Finished processing Named Entities and Punctuation.")

    return labels
