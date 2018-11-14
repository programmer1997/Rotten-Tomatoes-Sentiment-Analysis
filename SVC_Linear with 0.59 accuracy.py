# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv("../input/train.tsv", sep="\t")
print(df.shape)
# Any results you write to the current directory are saved as output.
X = df.Phrase
y = df.Sentiment

test_df = pd.read_csv("../input/test.tsv", sep='\t')
X_test=test_df.Phrase
X_test_PhraseID=test_df.PhraseId

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC


#Pipeline assemble the steps for feature extraction, tf-idf weighting and linear SVC
pipeline = Pipeline([
    ('vect', CountVectorizer()),    #Used for tokenisation and representing term frequency as a document-term matrix
    ('tfidf', TfidfTransformer()),  #Used for applying tf-idf term weighting scheme on the document-term matrix
    ('clf', LinearSVC()),           #Linear Support Vector Classifier that implements one-vs-rest scheme(OVR)
])


#Parameter for hyperparameter tunning
param_grid = {
    'vect__max_df':[0.8,0.9,1.0],
    'clf__C':[0.01,0.1,1.0]

}

#Performs exhaustive search over the specified parameter values and find the estimator that gives the best performance
#3-fold cross validation is performed to reduce the occurrence of overfitting
grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid)

#Fit data into the model
gridSearch = grid.fit(X, y)
print("Best: %f using %s" % (grid.best_score_, grid.best_params_)) #mean cross validated score for best estimator



skf = StratifiedKFold(n_splits=3)

for train, test in skf.split(X, y):
    grid.fit(X[train], y[train])
    train_score = gridSearch.best_estimator_.score(X[train], y[train])
    test_score = gridSearch.best_estimator_.score(X[test], y[test])
    print("Train = {}, Test = {}".format(train_score, test_score))
    
f = open('../SampleSubmission.csv', 'w')
f.write('PhraseId,Sentiment\n')

#Generate predicted labels
predicted_classes = gridSearch.best_estimator_.predict(X_test)
for i in range(0,X_test_PhraseID.shape[0]):
    f.write(str(X_test_PhraseID[i])+","+str(predicted_classes[i])+'\n')

f.close()
