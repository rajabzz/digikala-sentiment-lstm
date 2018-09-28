import json
import random
import re
import pickle

import numpy as np
from keras.layers import Dense, Embedding, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from hazm import word_tokenize, Normalizer
from constants import *

random.seed(42)

normalizer = Normalizer()


def filter_data(filepath, filename):
    with open('{}/{}'.format(filepath, filename), 'r', encoding='utf8') as f:
        products = []
        for row in f.readlines():
            raw_data = json.loads(row)
            comments = raw_data.get('cmts', None)
            rate = raw_data.get('r', None)
            cat = raw_data.get('c', None)
            if comments is None or len(comments) == 0 or cat is None or rate is None:
                continue
            valid_comments = []
            for comment in comments:
                pol = comment.get('pol', None)
                if pol is not None and pol != 0:
                    valid_comments.append(comment)
            if len(valid_comments) == 0:
                continue
            raw_data['cmts'] = valid_comments
            products.append(raw_data)
        return products


def tokenize_text(text):
    text = text.replace('.', ' ')
    text = re.sub('\s+', ' ', text).strip()
    text = text.replace('\u200c', ' ').replace('\n', '').replace('\r', '').replace('ي', 'ی').replace('ك', 'ک')
    normalized_text = normalizer.normalize(text)
    tokens = word_tokenize(normalized_text)
    return tokens


def process_data(products):
    categories_set = set()
    all_comments = []
    for product in products:
        product_category = product.get('c', None)
        categories_set.add(product_category)
        comments = product.get('cmts', [])
        for comment_dict in comments:
            pol = comment_dict.get('pol', None)
            if pol is None:
                print('err')
            if pol == -1:
                pol = 0
            text = comment_dict.get('txt', '')
            if text is None:
                text = ''
            tokens = tokenize_text(text)
            all_comments.append({
                'pol': pol,
                'tokens': tokens
            })
    return all_comments


def prepare_training_data(processed_comments, word_idx):
    X = []
    y = []
    for comment in processed_comments:
        X.append([word_idx[token] for token in comment['tokens']])
        y.append(comment['pol'])
    return np.asarray(X), np.asarray(y)


def create_word_set(comments):
    word_set = set()
    for comment in comments:
        for token in comment['tokens']:
            word_set.add(token)
    return word_set


def create_word_index(words_iterable):
    result = dict()
    i = 1
    for w in words_iterable:
        result[w] = i
        i += 1
    result['UNK'] = i
    return result


def create_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size + 1, 128))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model


if is_training_data_ready:
    with open('processed_data.pickle', 'rb') as f:
        X, y, word_idx = pickle.load(f)
else:
    print('Filtering data...')
    products = filter_data(DATA_PATH, 'comments_3.jl')

    print('Processing data...')
    all_comments = process_data(products)

    print('Create word set...')
    word_set = create_word_set(all_comments)

    print('Create word to index...')
    word_idx = create_word_index(word_set)

    print('Prepare training data...')
    X, y = prepare_training_data(all_comments, word_idx)

    with open('processed_data.pickle', 'wb') as f:
        pickle.dump((X, y, word_idx), f)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = sequence.pad_sequences(X_train, maxlen=MAX_LENGTH_OF_COMMENT)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LENGTH_OF_COMMENT)

if is_data_model_ready:
    model = load_model('models/model.h5')
else:
    model = create_model(len(word_idx))
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=2,
              validation_data=(X_test, y_test))
    model.save('models/model.h5')


y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

acc_sum = 0

real_count = [0, 0]
pred_count = [0, 0]
true_count = [0, 0]

for i in range(y_pred.shape[0]):
    label = y_test[i]
    pred = y_pred[i]
    plabel = -1

    if pred[label] > pred[1 - label]:
        plabel = label
    else:
        plabel = 1 - label

    real_count[label] += 1
    pred_count[plabel] += 1

    if label == plabel:
        acc_sum += 1
        true_count[label] += 1


print('acc', acc_sum / y_pred.shape[0])
print(real_count)
print(pred_count)
print(true_count)

p_negative = true_count[0] / pred_count[0]
p_positive = true_count[1] / pred_count[1]

r_negative = true_count[0] / real_count[0]
r_positive = true_count[1] / real_count[1]

print("p-", p_negative)
print("p+", p_positive)
print("r-", r_negative)
print("r+", r_positive)

f1_negative = 2 * (p_negative * r_negative) / (p_negative + r_negative)
f1_positive = 2 * (p_positive * r_positive) / (p_positive + r_positive)

print("f1-", f1_negative)
print("f1+", f1_positive)

print('>>> Interactive mode')
while True:
    text = input('comment: ')
    tokens = tokenize_text(text)
    tokens_idx = [[word_idx.get(token, word_idx['UNK']) for token in tokens]]
    X_interactive = sequence.pad_sequences(tokens_idx, maxlen=MAX_LENGTH_OF_COMMENT)
    result = model.predict(X_interactive)
    print(' - : ', str(round(result[0][0] * 100, 4)) + '%')
    print(' + : ', str(round(result[0][1] * 100, 4)) + '%', '\n')


