import pandas as pd
from preprocess_nlp import preprocess
from preprocess_nlp import glove

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout


def main():
    df_train = pd.read_csv('nlp_data/Corona_NLP_train.csv', encoding='latin-1')
    df_test = pd.read_csv('nlp_data/Corona_NLP_test.csv', encoding='latin-1')

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
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation="softmax"))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, epochs = 5, batch_size=32, verbose = 'auto')

    print(model.evaluate(X_test, Y_test, verbose=0))

    Y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
    plt.show()

if __name__ == '__main__':
    main()


# TODO's:
### - CLEAN UP CODE
### - WHAT LSTM ARCHITECTURE IS SUITABLE FOR THIS DATA?
### - MORE PREPREPROCESSING FOR DATA - currently, emojis are not removed, maybe some more noise is still captured.
### - WHAT PADDING IS APPROPRIATE, WHAT LENGTH SHOULD BE TAKEN?