import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils


def loadTestData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    X_test=np.array(list(D['Phrase']))
    X_test_PhraseID=np.array(list(D['PhraseId']))
    return  X_test,X_test_PhraseID
    
def labelize_movie_ug(movie,label):
    result = []
    prefix = label
    for i, t in zip(movie.index, movie):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

#Defining headers for preprocessing
cols = ['phraseID','sentenceID','phrase','sentiment']
df = pd.read_table("train.tsv",sep='\t',header=1, names=cols)



# above line will be different depending on where you saved your data, and your file name
df.drop(['phraseID','sentenceID'],axis=1,inplace=True)

print(df)


df['pre_clean_len'] = [len(t) for t in df.phrase]

x = df.phrase
y = df.sentiment

#Splitting the dataset
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)
print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% somewhat negative, {3:.2f}% neutral, {4:.2f}% positive, {5:.2f}% positive".format(len(x_train),(len(x_train[y_train == 0]) / (len(x_train)*1.))*100,(len(x_train[y_train == 1]) / (len(x_train)*1.))*100,(len(x_train[y_train == 2]) / (len(x_train)*1.))*100,(len(x_train[y_train == 3]) / (len(x_train)*1.))*100,(len(x_train[y_train == 4]) / (len(x_train)*1.))*100))
print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% somewhat negative, {3:.2f}% neutral, {4:.2f}% positive, {5:.2f}% positive".format(len(x_validation),(len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,(len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100,(len(x_validation[y_validation == 2]) / (len(x_validation)*1.))*100,(len(x_validation[y_validation == 3]) / (len(x_validation)*1.))*100,(len(x_validation[y_validation == 4]) / (len(x_validation)*1.))*100))
print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% somewhat negative, {3:.2f}% neutral, {4:.2f}% positive, {5:.2f}% positive".format(len(x_test),(len(x_test[y_test == 0]) / (len(x_test)*1.))*100,(len(x_test[y_test == 1]) / (len(x_test)*1.))*100,(len(x_test[y_test == 2]) / (len(x_test)*1.))*100,(len(x_test[y_test == 3]) / (len(x_test)*1.))*100,(len(x_test[y_test == 4]) / (len(x_test)*1.))*100))


X_test,X_test_PhraseID = loadTestData('test.tsv')

all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v = labelize_movie_ug(all_x, 'all')

cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha

model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha


embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))


#Tokenizing the phrases into nmbers
Tokenizer = Tokenizer(num_words=18000)
Tokenizer.fit_on_texts(np.concatenate((x_train, X_test), axis=0))
Tokenizer_vocab_size = len(Tokenizer.word_index) + 1
Tokenizer.fit_on_texts(x_train)
sequences = Tokenizer.texts_to_sequences(x_train)

#Padding all phrases to same length
x_train_seq = pad_sequences(sequences, maxlen=57)
print('Shape of data tensor:', x_train_seq.shape)


sequences_val = Tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=57)

encoded_Xtest = Tokenizer.texts_to_sequences(X_test)
X_test_encodedPadded = pad_sequences(encoded_Xtest, maxlen=57)

num_words = 100000
embedding_matrix = np.zeros((num_words, 200))
for word, i in Tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        

seed = 7


y_train = keras.utils.to_categorical(y_train,num_classes=5)
y_validation = keras.utils.to_categorical(y_validation,num_classes=5)

model_cnn_03 = Sequential()

e = Embedding(100000, 200, weights=[embedding_matrix], input_length=57, trainable=True)
model_cnn_03.add(e)
model_cnn_03.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='softmax', strides=1))
model_cnn_03.add(GlobalMaxPooling1D())
model_cnn_03.add(Dense(256, activation='relu'))
model_cnn_03.add(Dense(5, activation='softmax'))
model_cnn_03.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn_03.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=1, batch_size=32, verbose=2)

print("Predicting")

f = open('Submission.csv', 'w')
f.write('PhraseId,Sentiment\n')

predicted_classes = model_cnn_03.predict_classes(X_test_encodedPadded, batch_size=32, verbose=2)
for i in range(0,X_test_PhraseID.shape[0]):
    f.write(str(X_test_PhraseID[i])+","+str(predicted_classes[i])+'\n')

f.close()

print("Done Predicting")
