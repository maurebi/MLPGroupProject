import spacy
import string

nlpEN = spacy.load('en_core_web_sm')

# def ne_punct_recognizer(words, tweet_num, tweets):
#     print(f"- Processing Named Entities and Punctuation...")
#     labels = []
#     currentTweet = None
#     nlpEN = spacy.load('en_core_web_sm')
#     punctuation_set = set(string.punctuation)

#     for i in range(len(words)): 
#         # Keep in mind what tweet we are
#         if tweet_num[i] != currentTweet:
#             currentTweet = str(tweets[int(tweet_num[i])])
#             doc = nlpEN(currentTweet)
#             entity_words = {ent.text for ent in doc.ents}
        
#         if words[i] in punctuation_set:
#             labels.append('other')
#             continue
        
#         if entity_words and any(words[i] in ent.split() for ent in entity_words):
#             labels.append('ne')
#         else:
#             labels.append('none')
#     print(f"- Finished processing Named Entities and Punctuation...") 


#     return labels

def ne_punct_recognizer(words, tweet_num, tweets):
    print(f"- Processing Named Entities and Punctuation...")
    labels = []
    currentTweet = None
    punctuation_set = set(string.punctuation)

    for i in range(len(words)): 
        # Keep in mind what tweet we are
        if tweet_num[i] != currentTweet:
            currentTweet = str(tweets[int(tweet_num[i])])
            doc = nlpEN(currentTweet)  # <--- use the preloaded model!
            entity_words = {ent.text for ent in doc.ents}
        
        if words[i] in punctuation_set:
            labels.append('other')
            continue
        
        if entity_words and any(words[i] in ent.split() for ent in entity_words):
            labels.append('ne')
        else:
            labels.append('none')
    print(f"- Finished processing Named Entities and Punctuation...") 

    return labels
