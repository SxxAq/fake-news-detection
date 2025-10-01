import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()


def clean_text(text):
  """Cleans a single text:
  
    - Removes URLs
    - Removes non alpha-neumric characters
    - Converts to lowercase
    
  """
  text=re.sub(r"http\S+|www\S+", "",str(text))
  text=re.sub(r"[^a-zA-Z]", " ",text)
  text=text.lower()
  return text

def tokenize(text):
  """Tokenizes text into words"""
  return word_tokenize(text)


def remove_stopwords(tokens):
  """Removes common stopwords"""
  return [word for word in tokens if word not in stop_words]


def lemmatize_tokens(tokens):
  """Convertts token to their lemma (base forms)"""
  return [lemmatizer.lemmatize(word) for word in tokens]


def preprocess_text(text):
  """"
  Full preprocessing pipeline:
    1. Clean text
    2. Tokenize
    3. Remove stopwords
    4. Lemmatize
    5. Return final string 
    
  """
  text=clean_text(text)
  tokens=tokenize(text)
  tokens=remove_stopwords(tokens)
  tokens=lemmatize_tokens(tokens)
  return " ".join(tokens)
  
  

