import numpy as np
import bcolz
import pickle

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'6B.100.dat', mode='w')

with open(f'glove.6B.100d.txt.word2vec', 'rb') as f:
    for l in f:
        if idx == 0:
            idx += 1
            continue
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400000, 100)), rootdir=f'6B.100.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'6B.100_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'6B.100_idx.pkl', 'wb'))