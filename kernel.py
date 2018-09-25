#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import re
from datetime import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[6]:

train = pd.read_csv("D:\\data\\movies_reviews\\train.tsv", sep="\t")
test = pd.read_csv("D:\\data\\movies_reviews\\test.tsv", sep='\t')

# In[7]:


train = train[["Phrase", "Sentiment"]]
train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())



max_fatures = 100
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train['Phrase'].values)
X = tokenizer.texts_to_sequences(train['Phrase'].values)
X = pad_sequences(X)
Y = pd.get_dummies(train['Sentiment']).values

# In[13]:


####### BUILD model ##########
embed_dim = 128
lstm_out = 196

# model = Sequential()
# model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
# model.add(SpatialDropout1D(0.4))
# model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(5,activation='softmax'))
# model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

model = load_model('model25Sep18-010900')
print(model.summary())


# In[14]:


####### convert to 1-hot vector #######

checkpoint = ModelCheckpoint("model" + "{:%d%b%y-%H%M%S}".format(datetime.now()) + ".h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)


# In[25]:


batch_size = 32
model.fit(X, Y, validation_split=0.20, epochs = 60, batch_size=batch_size, verbose = 1, callbacks=[checkpoint])


# In[26]:


test = test[["PhraseId", "Phrase"]]
test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())
X_test = tokenizer.texts_to_sequences(test['Phrase'].values)
X_test = pad_sequences(X_test, maxlen=X.shape[1])
result = model.predict(X_test, batch_size=32, verbose=1)
predictions = [np.argmax(line) for line in result]
res = pd.DataFrame({"PhraseId": test["PhraseId"], "Sentiment": predictions})
res.to_csv("submit.csv", index=False)


# In[ ]:




