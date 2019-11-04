# coding: utf-8
import numpy as np
from keras import models

from train import load_data


def main():
    data, words = load_data()
    model = models.load_model('model.couplets.h5')
    c = 13
    x = np.zeros((1, 2 * c), dtype='uint32')
    for i in range(2 * c):
        s = max(i - c, 0)
        probas = model.predict(x[:, s:s + c], verbose=0)
        probas = probas.astype('float64')
        probas /= probas.sum()
        probas = np.random.multinomial(1, probas[0], 1)
        char = np.argmax(probas)
        x[0, i] = char
        char = words[char]
        print(char)


main()
