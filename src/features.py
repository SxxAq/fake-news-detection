from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

def get_bow_features(texts,max_features=5000):
  """
    Convert list of texts into Bag-of-Words features.
    
    Args:
        texts: list or pandas Series of strings
        max_features: maximum number of words to consider
    
    Returns:
        X: BoW feature matrix
        vectorizer: trained CountVectorizer object
  """
  
  vectorizer=CountVectorizer(max_features=max_features)
  X=vectorizer.fit_transform(texts)
  return X,vectorizer

def get_tfidf_features(texts,max_features=5000,ngram_range=(1,2)):
  """
    Convert list of texts into TF-IDF features.
    
    Args:
        texts: list or pandas Series of strings
        max_features: maximum number of words/features
        ngram_range: tuple for n-grams, e.g., (1,2) for unigrams + bigrams
    
    Returns:
        X: TF-IDF feature matrix
        vectorizer: trained TfidfVectorizer object
  """
  vectorizer=TfidfVectorizer(max_features=max_features,ngram_range=ngram_range)
  X=vectorizer.fit_transform(texts)
  return X,vectorizer
