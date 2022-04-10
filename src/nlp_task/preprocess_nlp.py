import re
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from gensim.scripts.glove2word2vec import glove2word2vec

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4')

def preprocess(df_train, df_test):
    df_train = df_train.drop(labels = ['UserName', 'ScreenName', 'Location', 'TweetAt'], axis = 1)
    df_test = df_test.drop(labels = ['UserName', 'ScreenName', 'Location', 'TweetAt'], axis = 1)

    le = preprocessing.LabelEncoder()

    df_train['Sentiment'] = le.fit_transform(df_train['Sentiment'])
    df_test['Sentiment'] = le.fit_transform(df_test['Sentiment'])
    # df_train['Sentiment'] = [label+2 if label == 0 else label for label in df_train['Sentiment']]
    # df_train['Sentiment'] = [label+3 if label == 1 else label for label in df_train['Sentiment']]
    # df_test['Sentiment'] = [label+2 if label == 0 else label for label in df_test['Sentiment']]
    # df_test['Sentiment'] = [label+3 if label == 1 else label for label in df_test['Sentiment']]


    df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(lambda x: clean_text(x)) 
    df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(lambda x: clean_text(x))

    X_train, X_val, Y_train, Y_val = train_test_split(np.asarray(df_train['OriginalTweet']), np.asarray(df_train['Sentiment']), train_size = 0.9) # split train into 90% train, 10% validation
    
    X_test = np.asarray(df_test['OriginalTweet'])
    Y_test = np.asarray(df_test['Sentiment'])
    
    ohe = preprocessing.OneHotEncoder()

    Y_train = ohe.fit_transform(Y_train.reshape(-1,1)).toarray()
    Y_val = ohe.fit_transform(Y_val.reshape(-1,1)).toarray()
    Y_test = ohe.fit_transform(Y_test.reshape(-1,1)).toarray()

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# This function removes all redundant information from tweets, removes stopwords and transforms each word into its lemma.
def clean_text(text):
    text = "".join(re.sub("(#[A-Za-z0-9_]+)"," ", text)) # remove hashtags
    text = "".join(re.sub("(@[A-Za-z0-9_]+)"," ", text)) # remove mentions
    text = "".join(re.sub("http\S+", " ", text)) # remove links
    text = "".join(re.sub("[^\x00-\x7f]"," ", text)) # remove weird ascii values
    text = "".join(re.sub("[0-9]+"," ", text)) # remove digits
    text = "".join(re.sub("[^\w\s]", " ", text)) # remove punctuation
    text = " ".join(text.split()) # remove redundant whitespace
    text = text.lower()
    text = text.strip() # remove leading and trailing whitespace

    tokens = word_tokenize(text)
    stops = set(stopwords.words("english")) # performance optimization
    token_list = [WordNetLemmatizer().lemmatize(token) for token in tokens if token not in stops] # remove stopwords and take lemma
    text = " ".join(token_list)
    return text

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
    f = open('nlp_data/glove.twitter.27B.200d.txt')
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
