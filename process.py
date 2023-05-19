import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import PyPDF2
import spacy

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

nlp = spacy.load("it_skills_ner")

def preprocess(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    stop_words = set(stopwords.words("russian"))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    gen = ""
    for i in tokens:
      gen = gen + i + " "
    return gen



def extract_skills(text):
    it_array = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "IT-SKILL":
            it_array.append(ent.text)
    return it_array

