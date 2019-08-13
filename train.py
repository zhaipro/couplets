# coding: utf-8
import csv
import os
import pathlib

import numpy as np
import keras
from keras import models
from keras import layers
from keras.callbacks import ReduceLROnPlateau


def _build(fn_csv, fn_npz, maxlen=13):
    # 读取，并过滤
    cr = csv.reader(open(fn_csv, encoding='utf-8-sig'))
    couplets = []
    for first, second in cr:
        if len(first) == len(second) and len(first) <= maxlen:
            couplets.append((first, second))
    # 张量化
    words = [' ']
    words.extend(set(word for first, second in couplets for word in first + second))
    word_int_map = dict(zip(words, range(len(words))))
    n = len(couplets)
    data = np.zeros((n, 2, maxlen), dtype=np.int32)
    for i, (first, second) in enumerate(couplets):
        for j, word in enumerate(first):
            data[i, 0, j] = word_int_map[word]
        for j, word in enumerate(second):
            data[i, 1, j] = word_int_map[word]
    np.savez(fn_npz, data=data, words=words)


def load_data(fn='data.csv'):
    # 下载
    url = 'https://raw.githubusercontent.com/oyrx/SpringCoupletData/master/SpringCouplets.csv'
    fn_csv = keras.utils.get_file(fn, url, cache_dir='.')
    # 构建
    fn_npz = pathlib.Path(fn_csv).with_suffix('.npz')
    if not os.path.isfile(fn_npz):
        _build(fn_csv, fn_npz)
    # 加载
    data = np.load(fn_npz)
    return data['data'], data['words']


def to_string(words, couplet):
    first = ''.join(words[couplet[0]])
    second = ''.join(words[couplet[1]])
    return f'{first}{second}'


def train():
    data, words = load_data()
    n, _, c = data.shape
    x_data = np.zeros((n * 2 * c, c), dtype='uint32')
    y_data = np.zeros(n * 2 * c, dtype='uint32')
    data.shape = n, 2 * c
    for i in range(2 * c):
        x_data[n * i:n * (i + 1), :min(i, c)] = data[:, max(0, i - c):i]
        y_data[n * i:n * (i + 1)] = data[:, i]

    model = models.Sequential([
        layers.Embedding(words.size, 32, input_length=c),
        layers.LSTM(32, dropout=0.75, recurrent_dropout=0.1),
        layers.Dense(words.size, activation='softmax'),
    ])
    reduce_lr = ReduceLROnPlateau(verbose=1)
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    model.summary()
    model.fit(x_data, y_data,
              epochs=100, callbacks=[reduce_lr])
    x = np.zeros((1, 2 * c), dtype='uint32')
    for i in range(2 * c):
        s = max(i - c, 0)
        probas = model.predict(x[s:s + c], verbose=0)
        probas = np.random.multinomial(1, probas, 1)
        char = np.argmax(probas)
        x[i] = char
        char = words[char]
        print(char)


train()
