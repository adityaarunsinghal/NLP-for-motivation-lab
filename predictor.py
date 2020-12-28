from scipy.sparse import coo_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
from nltk.corpus import stopwords

class predictor:
    def __init__(self, df, Xcolname, kind='l2', solver='lbfgs'):
        self.stop_words = stopwords.words("english")
        df['num_stop'] = df[Xcolname].apply(self.count_stop)
        self.df = df
        self.kind=kind
        self.solver=solver
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words={'english'})
        X = self.vectorizer.fit_transform(df[Xcolname])
        self.X = hstack((X,np.array(df['num_stop'])[:,None]))

    def get_X(self):
        return (self.X)
    
    def count_stop(self, entry):
        new = entry.lower()
        new = nltk.word_tokenize(new)
        num_stop = sum([1 for x in new if x in self.stop_words])
        return(num_stop)

    def set_Y(self, Ycolname):
        self.y = self.df[Ycolname]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.10)
        
    def train_on(self, Ycolname):
        self.set_Y(Ycolname)
        self.model = LogisticRegression(max_iter=1000, penalty=self.kind, solver=self.solver, class_weight='balanced').fit(self.x_train,self.y_train)
        print(cross_validate(self.model, self.x_train, self.y_train, return_train_score=True, cv=3))

    def manual_X_y(self, X, y):
        self.X = X
        self.y = y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.15)
        self.model = LogisticRegression(max_iter=1000, penalty=self.kind, solver=self.solver, class_weight='balanced').fit(self.x_train,self.y_train)
        print(cross_validate(self.model, self.x_train, self.y_train, return_train_score=True, cv=3))

    def test(self):
        score = self.model.score(self.x_test, self.y_test)
        print(score)
        return(score)
    
    def predict(self, x):
        count = self.count_stop(x[0])
        x = self.vectorizer.transform(x)
        x = hstack((x,np.array([count])[:,None]))
        return(self.model.predict_proba(x), "The pred is = ", self.model.predict(x))