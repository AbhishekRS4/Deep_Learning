import pandas as pd
import numpy as np
import seaborn as sns

import emoji
import re
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cleaner as used in similar BERT application (source: https://www.kaggle.com/code/edgardjonathan/bert-deep-learning)
def clean_text(tweet):
    # remove links
    tweet = "".join(re.sub("(\w+:\/\/\S+)", " ", tweet))

    # remove hashtags
    tweet = "".join(re.sub("(#[A-Za-z0-9_]+)", " ", tweet))

    # remove user mention
    tweet = "".join(re.sub("(@[A-Za-z0-9_]+)", " ", tweet))

    # remove none alphanumeric and aposthrope
    tweet = "".join(re.sub("([^0-9A-Za-z \t'])", " ", tweet))

    # remove extra whitespace
    tweet = " ".join(tweet.split())

    # remove emoji unicode
    tweet = "".join(c for c in tweet if c not in emoji.UNICODE_EMOJI)  # Remove Emojis

    # remove leading and trailing space
    tweet = tweet.strip()

    return tweet

def preprocess(df_train, df_test):
    # remove redundant columns
    df_train = df_train.drop(labels = ['UserName', 'ScreenName', 'Location', 'TweetAt'], axis = 1)
    df_test = df_test.drop(labels = ['UserName', 'ScreenName', 'Location', 'TweetAt'], axis = 1)

    le = preprocessing.LabelEncoder()

    # transform to integer labels
    df_train['Sentiment'] = le.fit_transform(df_train['Sentiment'])
    df_test['Sentiment'] = le.fit_transform(df_test['Sentiment'])

    # pply cleaning of tweets
    df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(lambda x: clean_text(x)) 
    df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(lambda x: clean_text(x))

    X_train, X_val, Y_train, Y_val = train_test_split(np.asarray(df_train['OriginalTweet']), np.asarray(df_train['Sentiment']), train_size = 0.9) # split train into 90% train, 10% validation
    
    X_test = np.asarray(df_test['OriginalTweet'])
    Y_test = np.asarray(df_test['Sentiment'])
    
    ohe = preprocessing.OneHotEncoder()

    # apply one-hot encoding to the labels
    Y_train = ohe.fit_transform(Y_train.reshape(-1,1)).toarray()
    Y_val = ohe.fit_transform(Y_val.reshape(-1,1)).toarray()
    Y_test = ohe.fit_transform(Y_test.reshape(-1,1)).toarray()

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def glove(X_train, X_test):
    t = Tokenizer(oov_token = True)
    t.fit_on_texts(X_train) # make up an internal dictionary of training vocabulary

    vocab_size = len(t.word_index) + 1
    # map both datasets to the integer representation of the dictionary
    X_train_final = t.texts_to_sequences(X_train)
    X_test_final = t.texts_to_sequences(X_test)

    # pad sequences to have uniform size
    X_train_final = pad_sequences(X_train_final, maxlen=45, padding = "post")
    X_test_final = pad_sequences(X_test_final, maxlen=45, padding = "post")

    # create dictionary that contains the vector representation of each word present in the GloVe set
    f = open('glove.twitter.27B.200d.txt')
    embeddings_index = dict()
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # create embedding matrix that contains all the vector representations for each word, ordered by their Tokenizer index
    embedding_matrix = np.zeros((vocab_size, 200))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, X_train_final, X_test_final, vocab_size

def main():
    df_train = pd.read_csv('Corona_NLP_train.csv', encoding='latin-1')
    df_test = pd.read_csv('Corona_NLP_test.csv', encoding='latin-1')

    X_train, X_val, X_test, Y_train, Y_val, Y_test = preprocess(df_train, df_test)

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("Y_train:", Y_train.shape)
    print("Y_val:", Y_val.shape)
    print("Y_test:", Y_test.shape)

    embedding_matrix, X_train, X_test, vocab_size = glove(X_train, X_test)
    model = Sequential()
    model.add(Embedding(vocab_size, 200, weights = [embedding_matrix], input_length = 45, trainable = False))
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(5, activation="softmax"))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    print(model.summary())

    model.fit(X_train, Y_train, epochs = 5, batch_size=32, verbose = 'auto')
    print(model.evaluate(X_test, Y_test, verbose=0))

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.argmax(axis=1), Y_pred.argmax(axis=1)))

    cf_matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
    cf_matrix = pd.DataFrame(cf_matrix)

    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.6)
    sns.heatmap(cf_matrix, annot=True, fmt="g")
    plt.savefig('confusion_matrix_baseline.png', bbox_inches='tight')
    
if __name__ == '__main__':
    main()