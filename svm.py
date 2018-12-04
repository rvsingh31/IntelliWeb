import codecs
import pandas as pd
from bs4 import BeautifulSoup
import string
import numpy as np

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

lemmatizer = WordNetLemmatizer()
# lemmatizer = WordNetLemmatizer()
stopset = set(stopwords.words('english'))

def main():
    # f = codecs.open("example.html", 'r')
    # f1 = codecs.open("example1.html", 'r')
    # f2 = codecs.open("example2.html",'r')
    # f3 = codecs.open("example3.html",'r')
    # data = [['course', f1],['course',f],['department',f2],['department',f3]]
    # df = pd.DataFrame(data, columns=['Class', 'text'])
    data = pd.read_csv("raw.csv")
    df = pd.DataFrame(data, columns=['text', 'Class'])
    df = df[df['Class'].isin([1,3,4,5,6])]
    df['prc'] = df['text'].apply(process)
    df['prc_conct'] = df['prc'].apply(lambda tokens: ' '.join(str(v) for v in tokens))
    X_tfidf = Vectorization(df['prc_conct'])
    y = df['Class']
    X = X_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    # scaler = MinMaxScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # clf1 = MultinomialNB()
    print("Going for Training")
    svm_clf = svm.SVC(100, kernel='linear')
    svm_clf.fit(X_train,y_train)
    # clf1.fit(X_train, y_train)
    y_true, y_pred = y_test, svm_clf.predict(X_test)
    # y_true, y_pred = y_test, clf1.predict(X_test)
    print(c_matrix(y_true, y_pred))
    # print(df.head(10))

def Vectorization(txt):
    print("tfid")
    vectorizer = TfidfVectorizer(max_features=1000)
    print("Vectorization")
    X_tfidf = vectorizer.fit_transform(txt)
    return X_tfidf

def c_matrix(y_true, y_pred, num_classes=5):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    cm_np = np.asarray(cm)
    TP = np.diag(cm_np)
#     print(TP)
    FP = np.sum(cm, axis=0) - TP
    print(FP)
    FN = np.sum(cm, axis=1) - TP
#     print(FN)
    TN = []
    for i in range(num_classes):
        temp = np.delete(cm, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
#     print(TN)
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)
    acc = (TP+TN)/(TP+FP+TN+FN)
    f1 = 2*prec*rec/(prec+rec)

    print("accuracy", acc)
    print("precision", prec)
    print("recall", rec)
    print("f1", f1)
    return prec, rec, acc, f1


# def process(txt):
#     cleantext = BeautifulSoup(txt, "lxml").text
#     # print(cleantext)
#     tokens = []
#     for token, tag in pos_tag(wordpunct_tokenize(cleantext)):
#         token = token.lower()
#         token = token.strip()  # Strip whitespace and other punctuations
#         token = token.strip('_')  # remove _ if any
#         token = token.strip('*')
#         if token in stopwords.words('english'):
#             continue
#         if all(char in string.punctuation for char in token):
#             continue
#
#         if token.isdigit():
#             continue
#         tokens.append(token)
#     x = [lemmatizer.lemmatize(word) for word in tokens]
#     return x

def process(txt):
    cleantext = BeautifulSoup(txt, "lxml").text
    tokens = []
    for token in wordpunct_tokenize(cleantext):
        if token.isdigit():
            continue
        if all(char in string.punctuation for char in token):
            continue

        token = token.lower()
        token = token.strip()  # Strip whitespace and other punctuations
        token = token.strip('_')  # remove _ if any
        token = token.strip('*')
        #         if token in stopset:
        #             continue
        tokens.append(token)
        lemmatizer.lemmatize(token)
    #     x = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

if __name__ == '__main__':
    main()
