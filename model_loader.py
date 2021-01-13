
from scipy.sparse import coo_matrix, hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk import word_tokenize
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from joblib import dump, load
import os

class model_loader:
    def __init__(self, modelname, path="/Users/aditya/Documents/GitHub/NLP-for-motivation-lab/models/"):
        folder = path+modelname+"/"
        self.path = folder
        self.stop_words = stopwords.words("english")
        self.model = load(f'{self.path}{modelname}_lr.joblib')
        self.vectorizer = load(f'{self.path}{modelname}_v.joblib')
        self.norm_extra1 = load(f'{self.path}{modelname}_norm1.joblib')
        self.norm_extra2 = load(f'{self.path}{modelname}_norm2.joblib')

    def count_stop(self, entry):
        new = entry.lower()
        new = word_tokenize(new)
        length = len(new)
        num_stop = sum([1 for x in new if x in self.stop_words])
        return(num_stop,length)

    def predict_values(self, x):
        df = pd.DataFrame(x)
        df['num_stop'], df['total_words'] = list(zip(*x.apply(self.count_stop)))
        df['num_stop'] = self.norm_extra1.transform(np.array(df['num_stop']).reshape(-1,1))
        df['total_words'] = self.norm_extra2.transform(np.array(df['total_words']).reshape(-1,1))
        self.X = self.vectorizer.transform(x)
        # print(self.X.shape)
        self.X = hstack([self.X, csr_matrix(np.array(df['num_stop'])).T])
        self.X = hstack([self.X, csr_matrix(np.array(df['total_words'])).T])
        # print(self.X.shape)
        return(self.model.predict(self.X))