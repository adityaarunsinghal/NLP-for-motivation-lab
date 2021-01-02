
from scipy.sparse import coo_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from joblib import dump, load

class model_loader:
    def __init__(self, modelname, path="/Users/aditya/Documents/GitHub/NLP-for-motivation-lab/models/"):
        self.path = path
        self.stop_words = stopwords.words("english")
        self.model = load(f'{self.path}{modelname}_lr.joblib')
        self.vectorizer = load(f'{self.path}{modelname}_v.joblib')

    def count_stop(self, entry):
        new = entry.lower()
        new = word_tokenize(new)
        length = len(new)
        num_stop = sum([1 for x in new if x in self.stop_words])
        return(num_stop,length)

    def predict_values(self, x):
        df = pd.DataFrame(x)
        df['num_stop'], df['total_words'] = list(zip(*x.apply(self.count_stop)))
        X = self.vectorizer.transform(x)
        self.X = hstack((X,np.array(df['num_stop'], df['total_words'])[:,None]))
        print(X.shape)
        return(self.model.predict(X))