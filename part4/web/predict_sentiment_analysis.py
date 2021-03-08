from nltk import data
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
import re
from nltk.util import everygrams
import pickle
from nltk.tokenize import word_tokenize
import json

from nltk import download
download()
download('punkt',download_dir=data.path[0])
download('stopwords',download_dir=data.path[0])
download('wordnet',download_dir=data.path[0])
# #THEN remove the zip files!

data.path=['nltk_data']
stopwords_eng = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def clean_words(words):
    return [w for w in words if w not in stopwords.words("english") and w not in punctuation]


def extract_features(document):
    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if w not in stopwords_eng and w not in punctuation]
    document = " ".join(lemmas)
    document = document.lower()
    document = re.sub(r'[^a-zA-Z0-9\s]', ' ', document)
    words = [w for w in document.split(" ") if w!="" and w not in stopwords_eng and w not in punctuation]
    return [str('_'.join(ngram)) for ngram in list(everygrams(words, max_len=3))]


def get_word_dict(words):
    words = clean_words(words)
    return dict([(w, True) for w in words])


model_file = open("sa_classifier.pickle", "rb")
model = pickle.load(model_file)
model_file.close()


def get_sentiment(review):
    words = extract_features(review)
    words = get_word_dict(words)
    return model.classify(words)


def predict(event,context):
    review = event['body']
    return { 'statusCode': 200, 'body': json.dumps(get_sentiment(review)) }

predict({"body":"This movie is amazing, with witty dialog and beautiful shots."}, None)
