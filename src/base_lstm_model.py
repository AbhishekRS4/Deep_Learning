import pandas as pd
from preprocess_nlp import preprocess
from preprocess_nlp import glove
def main():
    df_train = pd.read_csv('nlp_data/Corona_NLP_train.csv', encoding='latin-1')
    df_test = pd.read_csv('nlp_data/Corona_NLP_test.csv', encoding='latin-1')

    X_train, X_val, X_test, Y_train, Y_val, Y_test = preprocess(df_train, df_test)

    glove(X_train, X_val, X_test)


if __name__ == '__main__':
    main()