import os, glob, string, re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import metrics
from numpy.random import seed
from tensorflow import set_random_seed

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
seed(9)
set_random_seed(9)
lemmatizer = WordNetLemmatizer()

classes = ["student", "faculty", "staff", "department", "course", "project", "other"]
class_index = dict((c, i) for i, c in enumerate(classes))
stopset = set(stopwords.words('english'))
batch_size = 1000
initial_path = "/Users/saurabh/Downloads/ncsu/study/alda/ALDA-Project-Work/data/"


def get_all_files(path):
    all_files = {}
    all_folders = os.listdir(path)
    for clz in all_folders:
        if clz.startswith('.'):
            continue
        if clz not in all_files:
            all_files[clz] = []
        path_with_clz = path + clz + '/'
        all_univs = os.listdir(path_with_clz)
        for univ in all_univs:
            if univ.startswith('.'):
                continue
            path_with_univs = path_with_clz + univ + '/'
            all_files[clz].append(glob.glob(os.path.join(path_with_univs, '*')))
    return all_files


def get_raw_df(path, read_local):
    all_files = get_all_files(path)
    if not read_local:
        raw = []
        for k, v in all_files.items():
            for fnames in v:
                for fs in fnames:
                    with open(fs, 'rb') as f:
                        raw_data = f.read()
                        raw.append([raw_data, class_index[k]])

        raw_df = pd.DataFrame(raw, columns=["text", "Class"])
    else:
        raw_df = pd.read_csv('raw.csv')
    return raw_df


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
        if token in stopset:
            continue
        tokens.append(token)
        lemmatizer.lemmatize(token)
    return tokens


def c_matrix(y_true, y_pred, num_classes=7):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    cm_np = np.asarray(cm)
    TP = np.diag(cm_np)
    FP = np.sum(cm, axis=0) - TP
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


def generate_model(embedding_dim, embedding_matrix, max_length, vocab_size, y):
    model = Sequential()
    # model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(
        Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1]))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    return model


def generate_embeddings(glove_dir, vocab_size, word_index):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    # In the dataset, each line represents a new word embedding
    # The line starts with the word and the embedding values follow
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding
    f.close()
    all_embs = np.stack(embeddings_index.values())
    emb_mean = all_embs.mean()
    emb_std = all_embs.std()
    print(emb_mean, emb_std)
    embedding_dim = 100
    nb_words = min(vocab_size, len(word_index))  # How many words are there actually
    # Create a random matrix with the same mean and std as the embeddings
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_dim))
    # The vectors need to be in the same position as their index.
    # Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on
    # Loop over all words in the word index
    for word, i in word_index.items():
        # If we are above the amount of words we want to use we do nothing
        if i >= vocab_size:
            break
        # Get the embedding vector for the word
        embedding_vector = embeddings_index.get(word)
        # If there is an embedding vector, put it in the embedding matrix
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_dim, embedding_matrix


vocab_size = 100


def get_sequences(df):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df['processed_text_cnct'])
    sequences = tokenizer.texts_to_sequences(df['processed_text_cnct'])
    return sequences, tokenizer, vocab_size


def prepare_y_pred(y_pred):
    y_pred_mod = []
    for row in y_pred:
        max_val = max(row)
        mod_label = []
        for label_val in row:
            if label_val < max_val:
                mod_label.append(0.)
            else:
                mod_label.append(1.)
        y_pred_mod.append(mod_label)
    return y_pred_mod


def main():
    path = initial_path + "raw/webkb/"

    raw_df = get_raw_df(path, False)
    print("raw dataframe", raw_df.shape)
    # df = raw_df.sample(frac=0.9, replace=True)
    df = raw_df
    print("sample dataframe", df.shape)
    df['processed_text'] = df['text'].apply(process)
    df['processed_text_cnct'] = df['processed_text'].apply(lambda tokens: ' '.join(str(v) for v in tokens))

    sequences, tokenizer, vocab_size = get_sequences(df)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    avg = sum(map(len, sequences)) / len(sequences)
    std = np.sqrt(sum(map(lambda x: (len(x) - avg) ** 2, sequences)) / len(sequences))
    print("Tokens avg {} and std {}".format(avg, std))

    max_length = 100
    data = pad_sequences(sequences, maxlen=max_length)
    labels = to_categorical(np.asarray(df['Class']))
    print('Shape of data:', data.shape)
    print('Shape of labels:', labels.shape)
    glove_dir = initial_path + 'glove'

    embedding_dim, embedding_matrix = generate_embeddings(glove_dir, vocab_size, word_index)

    X = data
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = generate_model(embedding_dim, embedding_matrix, max_length, vocab_size, y)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1)

    y_pred = model.predict(X_test)
    y_pred_mod = prepare_y_pred(y_pred)

    c_matrix(y_test, np.array(y_pred_mod), num_classes=7)


if __name__ == '__main__':
    main()
