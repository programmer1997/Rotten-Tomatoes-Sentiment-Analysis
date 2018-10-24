import numpy as np
import pandas as pd
import os
#print(os.listdir("/drive/My Drive/Colab Notebooks/input"))

df = pd.read_csv("../input/train.tsv", sep="\t")
print(df.shape)

X = df.Phrase
y = df.Sentiment

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder



pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC()),
])



param_grid = {
    'vect__max_df':[0.7,0.8,0.9,1.0],
    'clf__C':[0.01,0.1,1.0]

}


grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid)


gridSearch = grid.fit(X, y)
print("Best: %f using %s" % (grid.best_score_, grid.best_params_)) #mean cross validated score for best estimator

ypred = cross_val_predict(gridSearch.best_estimator_, X, y)
#accuracy = accuracy_score(y, ypred)
print(classification_report(y, ypred, digits=3))