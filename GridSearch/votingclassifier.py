import pandas as pd
from bs4 import BeautifulSoup
import string

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import pos_tag
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

lemmatizer = WordNetLemmatizer()


def main():

    data = pd.read_csv("raw.csv")
    df = pd.DataFrame(data, columns=['text', 'Class'])
    df['prc'] = df['text'].apply(process)
    df['prc_conct'] = df['prc'].apply(lambda tokens: ' '.join(str(v) for v in tokens))
    X_tfidf = Vectorization(df['prc_conct'])
    y = df['Class']
    print(y)
    X = X_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    estimators = []
    clf1 = MultinomialNB()
    estimators.append(clf1)
    clf2 = svm.SVC(C=100, kernel='linear')
    estimators.append(clf2)
    eclf = VotingClassifier(estimators=[('nb', clf1), ('svm', clf2)], voting='hard')
    params = {'multinomialnb__C': [1.0, 100.0], 'svm_C': [1, 100] }
    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(eclf, param_grid=params, cv=5,scoring='%s_macro' % score)
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


def Vectorization(txt):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(txt)
    return X_tfidf


def process(txt):
    cleantext = BeautifulSoup(txt, "lxml").text
    tokens = []
    for token, tag in pos_tag(wordpunct_tokenize(cleantext)):
        token = token.lower()
        token = token.strip()
        token = token.strip('_')
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
