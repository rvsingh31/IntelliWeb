import codecs
import pandas as pd
from bs4 import BeautifulSoup
import string

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm

lemmatizer = WordNetLemmatizer()


def main():
    f = codecs.open("example.html", 'r')
    f1 = codecs.open("example1.html", 'r')
    f2 = codecs.open("example2.html",'r')
    f3 = codecs.open("example3.html",'r')
    data = [['course', f1],['course',f],['department',f2],['department',f3]]
    df = pd.DataFrame(data, columns=['Class', 'text'])
    df['prc'] = df['text'].apply(process)
    df['prc_conct'] = df['prc'].apply(lambda tokens: ' '.join(str(v) for v in tokens))
    X_tfidf = Vectorization(df['prc_conct'])
    y = df['Class']
    X = X_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    svm_clf = svm.SVC(100, kernel='linear')
    svm_clf.fit(X_train, y_train)
    y_true, y_pred = y_test, svm_clf.predict(X_test)
    print(confusion_matrix(y_true, y_pred))
    # print(df.head(10))

def Vectorization(txt):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(txt)
    return X_tfidf


def process(txt):
    cleantext = BeautifulSoup(txt, "lxml").text
    # print(cleantext)
    tokens = []
    for token, tag in pos_tag(wordpunct_tokenize(cleantext)):
        token = token.lower()
        token = token.strip()  # Strip whitespace and other punctuations
        token = token.strip('_')  # remove _ if any
        token = token.strip('*')
        if token in stopwords.words('english'):
            continue
        if all(char in string.punctuation for char in token):
            continue

        if token.isdigit():
            continue
        tokens.append(token)
    x = [lemmatizer.lemmatize(word) for word in tokens]
    return x


if __name__ == '__main__':
    main()
