import codecs
import pandas as pd
from bs4 import BeautifulSoup
import string
import numpy as np

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm

lemmatizer = WordNetLemmatizer()
stopset = set(stopwords.words('english'))

def main():
    data = pd.read_csv("raw.csv")
    df = pd.DataFrame(data, columns=['text', 'Class'])
    df['prc'] = df['text'].apply(process)
    df['prc_conct'] = df['prc'].apply(lambda tokens: ' '.join(str(v) for v in tokens))
    X_tfidf = Vectorization(df['prc_conct'])
    y = df['Class']
    X = X_tfidf

    k_fold = StratifiedKFold(n_splits=5)
    for train_index, test_index in k_fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svm_clf = svm.SVC(100, kernel='linear')
        svm_clf.fit(X_train, y_train)
        y_true, y_pred = y_test, svm_clf.predict(X_test)
        print(c_matrix(y_true, y_pred))

def Vectorization(txt):
    print("tfid")
    vectorizer = TfidfVectorizer(max_features=1000)
    print("Vectorization")
    X_tfidf = vectorizer.fit_transform(txt)
    return X_tfidf

def c_matrix(y_true, y_pred, num_classes=7):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    cm_np = np.asarray(cm)
    TP = np.diag(cm_np)
    FP = np.sum(cm, axis=0) - TP
    print(FP)
    FN = np.sum(cm, axis=1) - TP
    TN = []
    for i in range(num_classes):
        temp = np.delete(cm, i, 0)
        temp = np.delete(temp, i, 1)
        TN.append(sum(sum(temp)))
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    acc = (TP + TN) / (TP + FP + TN + FN)
    f1 = 2 * prec * rec / (prec + rec)

    print("accuracy", acc)
    print("precision", prec)
    print("recall", rec)
    print("f1", f1)
    return prec, rec, acc, f1


def process(txt):
    cleantext = BeautifulSoup(txt, "lxml").text
    tokens = []
    for token in wordpunct_tokenize(cleantext):
        if token.isdigit():
            continue
        if all(char in string.punctuation for char in token):
            continue

        token = token.lower()
        token = token.strip()
        token = token.strip('_')
        token = token.strip('*')
        tokens.append(token)
        lemmatizer.lemmatize(token)

    return tokens


if __name__ == '__main__':
    main()
