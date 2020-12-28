from scipy.sparse import coo_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import numpy as np
from nltk.corpus import stopwords
import imblearn
import random

class predictor:
    def __init__(self, df, Xcolname, kind='l2', solver='lbfgs', k_neighbors = 4):
        self.random_state = random.randint(0, 42)
        self.stop_words = stopwords.words("english")
        df['num_stop'], df['total_words'] = list(zip(*df[Xcolname].apply(self.count_stop))) # is adding total words a good idea? Need more dim reduction and proper feature engineering from code book AND a proper system to test performance
        self.df = df
        self.kind=kind
        self.solver=solver
        self.Xcolname = Xcolname
        self.balancer = imblearn.over_sampling.SMOTE(k_neighbors = k_neighbors, random_state=self.random_state) #need to make this more flexible to extremely underrepresented classes using randomoversampler
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words={'english'})
        X = self.vectorizer.fit_transform(df[Xcolname])
        self.X = hstack((X,np.array(df['num_stop'], df['total_words'])[:,None]))

    def get_X(self):
        return (self.X)
    
    def count_stop(self, entry):
        new = entry.lower()
        new = word_tokenize(new)
        length = len(new)
        num_stop = sum([1 for x in new if x in self.stop_words])
        return(num_stop,length)
        
    def train_on(self, Ycolname):
        self.y = self.df[Ycolname]
        self.X, self.y = self.balancer.fit_resample(self.X, self.y)
        self.split()
        self.make_model()

    def manual_X_y(self, X, y):
        self.X = X
        self.y = y
        self.split()
        self.make_model()

    def make_model(self):
        self.model = LogisticRegression(max_iter=1000, penalty=self.kind, solver=self.solver, class_weight='balanced').fit(self.x_train,self.y_train)
        print(cross_validate(self.model, self.x_train, self.y_train, return_train_score=True, cv=3))

    def split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.15)

    def test(self):
        score = self.model.score(self.x_test, self.y_test)
        print(score)
        return(score)
    
    def predict(self, x):
        count, length = self.count_stop(x)
        x = self.vectorizer.transform([x])
        # x = hstack((x,np.array([count, length])[:,None]))
        x = hstack((x,np.array([[count, length]])[:,None]))
        return(self.model.predict_proba(x) + "\n The predicted value is = " + self.model.predict(x))