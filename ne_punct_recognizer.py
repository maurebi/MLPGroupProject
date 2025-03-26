
def NE_and_punct_recognizer(words):
        try:
            # Try loading the Spanish model
            nlpEN = spacy.load("en_core_news_sm")
            nlpES = spacy.load("es_core_news_sm")
        except OSError:
            print("*** Spacy Model not yet found. Installing...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_news_sm"])
            subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
            nlpEN = spacy.load("en_core_news_sm")
            nlpES = spacy.load("es_core_news_sm")
            print("*** Installed EN and ES Spacy models ***")
            
        return