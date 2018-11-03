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

svc = LinearSVC(
    C=0.01,
    class_weight='balanced',
    dual=True,
    fit_intercept=True,
    intercept_scaling=1,
    loss='squared_hinge',
    max_iter=1000,
    multi_class='ovr',
    penalty='l2',
    random_state=0,
    tol=1e-05, 
    verbose=0
)

tfidf = CountVectorizer(
    input='content',
    encoding='utf-8',
    decode_error='strict',
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 1),
    analyzer='word',
    max_df=0.8,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
    dtype=np.int64
)

pipeline = Pipeline([
    ('tfidf', tfidf),   #fit and transform
    ('svc', svc),   
])

skf = StratifiedKFold(n_splits=3)

X = df.Phrase
y = df.Sentiment

for train, test in skf.split(X, y):
    pipeline.fit(X[train], y[train])
    train_score = pipeline.score(X[train], y[train])
    test_score = pipeline.score(X[test], y[test])
    print("Train = {}, Test = {}".format(train_score, test_score))
    
f = open('../input/submission.csv', 'w')
f.write('PhraseId,Sentiment\n')


predicted_classes = pipeline.predict(X_test)
for i in range(0,X_test_PhraseID.shape[0]):
    f.write(str(X_test_PhraseID[i])+","+str(predicted_classes[i])+'\n')

f.close()
