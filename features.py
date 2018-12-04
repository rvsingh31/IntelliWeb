import codecs
import pandas as pd
from bs4 import BeautifulSoup
import string
import numpy as np
import re
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
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
    df['title'] = df['text'].apply(extract_title)
    df['link'] = df['text'].apply(extract_links)
    df['links_and_title'] = df['title'] + df['link']
    df['data_and_title'] = df['prc_conct'] + df['title']
    df['data_and_link'] = df['prc_conct'] + df['link']
    df['data_and_title_and_link'] = df['prc_conct'] + df['links_and_title']
    # print (df['links_and_title'])

    # X_tfidf = Vectorization(df['prc_conct'])
    X_tfidf = Vectorization(df['data_and_title_and_link'],False)
    y = df['Class']
    X = X_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # GridSearch(X_train,X_test,y_train,y_test)

    # scaler = MinMaxScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # clf1 = MultinomialNB()
    #print("Going for Training")
    svm_clf = svm.SVC(10, kernel='linear')
    svm_clf.fit(X_train,y_train)
    # clf1.fit(X_train, y_train)
    y_true, y_pred = y_test, svm_clf.predict(X_test)
    # y_true, y_pred = y_test, clf1.predict(X_test)
    print(c_matrix(y_true, y_pred))
    # print(df.head(10))

    
def Vectorization(txt,flag = True):
    #print("tfid")
    if flag:
        vectorizer = TfidfVectorizer(max_features=100)
        print("Vectorization")
        X_tfidf = vectorizer.fit_transform(txt)
        return X_tfidf
    else:
        vectorizer = TfidfVectorizer(max_features = 100, decode_error='replace', encoding='utf-8')
        x = vectorizer.fit_transform(txt.values.astype('U'))
        return x

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

def process(txt,flag = True):

    if flag:
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

def extract_title(txt):
    soup = BeautifulSoup(txt,'lxml')
    title = ""
    try:
        title = soup.title.string
        title = title.replace("\\t","")
        title = title.replace("\\n","")
        title = title.replace("\\","")
        title = title.lower().strip()

    except:
        pass
    
    return title


def extract_links(txt):
    soup = BeautifulSoup(txt,'lxml')
    links = []
    try:
        for link in soup.find_all('a'):
            url = link.get('href')
            if url:
                temp = ' '.join(getWordsFromURL(url))
                links.append(temp)
    except:
        pass
    
    return ' '.join(links)

def getWordsFromURL(url):
    return re.compile(r'[~#\%+:./?=\-&^(0-9)]+',re.UNICODE).split(url)

def GridSearch(X_train,X_test,y_train,y_test):
    svm_clf = svm.SVC(100, kernel='linear')
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm_clf, tuned_parameters, cv=2,scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()



if __name__ == '__main__':
    main()
