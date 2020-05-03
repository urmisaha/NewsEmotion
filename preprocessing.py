'''
To run this file, two arguements are expected:
1. sampling: 'no-sample' or 'up-sample' or 'down-sample'
2. domain: 'restaurant' or 'movie' or 'music' or 'ride'
Command Example: python preprocessing.py no-sample restaurant

Read the lines from 55-58 and uncomment the correct line. Update the value of train_labels with the value of the train_sentences uncommented
Run the file with the proper argument depending on the line uncommented
'''
import sys
import bz2
from collections import Counter
import re
import nltk
import numpy as np
import pandas
import pickle
from sklearn.utils import resample

try:
    sampling = sys.argv[1]                              # takes values: no-sample | up-sample | down-sample
except:
    print("Error Message:\nArguement expected for sampling: no-sample | up-sample | down-sample")
    exit()

# domain = sys.argv[2]

dataframe = pandas.read_csv("dataset/train.csv", header=None, names=['sentence', 'sentiment'])
print(dataframe)

df_0 = dataframe[dataframe['sentiment']==0]
df_1 = dataframe[dataframe['sentiment']==1]
df_2 = dataframe[dataframe['sentiment']==2]
df_3 = dataframe[dataframe['sentiment']==3]
df_4 = dataframe[dataframe['sentiment']==4]
df_5 = dataframe[dataframe['sentiment']==5]
print(df_0.shape[0])
print(df_1.shape[0])
print(df_2.shape[0])
print(df_3.shape[0])
print(df_4.shape[0])
print(df_5.shape[0])
df_0 = resample(df_0, replace=True, n_samples=4000)
df_1 = resample(df_1, replace=True, n_samples=4000)
df_2 = resample(df_2, replace=True, n_samples=4000)
df_3 = resample(df_3, replace=True, n_samples=4000)
df_4 = resample(df_4, replace=True, n_samples=4000)
df_5 = resample(df_5, replace=True, n_samples=4000)
dataframe = pandas.concat([df_0, df_1, df_2, df_3, df_4, df_5])

# ======================================================================

train_size = dataframe.shape[0]
print(train_size)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
dataset = dataframe.values
train_sentences = dataset[0:24000,0]
train_labels = dataset[0:24000,1].astype(int)

print(train_sentences[0])

dataframe = pandas.read_csv("dataset/test.csv", header=None, names=['sentence', 'sentiment'])
dataframe = dataframe.dropna()
# if domain == "movie":
#     dataframe['sentiment'].replace(['positive', 'negative'], [1, 0], inplace=True)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
dataset = dataframe.values
test_sentences = dataset[0:1000,0]
test_labels = dataset[0:1000,1].astype(int)

print("Data load completed..")

# Some simple cleaning of data
for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])
for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d','0',test_sentences[i])

# Modify URLs to <url>
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

words = Counter()  # Dictionary that will map a word to the number of times it appeared in all the training sentences
# wordscount = []
# max_c = 0
# min_c = 1000000
for i, sentence in enumerate(train_sentences):
    # The sentences will be stored as a list of words/tokens
    train_sentences[i] = []
    sentence = sentence.replace("\\n", " ").replace("\\", "").replace("\/", "").replace("\\t", " ")
    tokens = nltk.word_tokenize(sentence)
    # max_c = max(max_c, len(tokens))
    # min_c = min(min_c, len(tokens))
    # wordscount.append(len(tokens))
    for word in tokens:   # Tokenizing the words
        words.update([word.lower()])            # Converting all the words to lowercase
        train_sentences[i].append(word)
    if i%20000 == 0:
        print(str((i*100)/train_size) + "% done")
print("100% done")
# print("max_c = " + str(max_c))
# print("min_c = " + str(min_c))
# print("average length = " + str(sum(wordscount)/len(wordscount)))

# Removing the words that only appear once
# words = {k:v for k,v in words.items() if v>1}
words = {k:v for k,v in words.items()}

# Sorting the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)

# Adding padding and unknown to our vocabulary so that they will be assigned an index
words = ['_PAD','_UNK'] + words

# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:(i+1) for i,o in enumerate(words)}
idx2word = {(i+1):o for i,o in enumerate(words)}

for i, sentence in enumerate(train_sentences):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

for i, sentence in enumerate(test_sentences):
    # For test sentences, we have to tokenize the sentences as well
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

seq_len = 300  # The length that the sentences will be padded/shortened to

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)

# Converting our labels into numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

split_frac = 0.5    # 50% validation, 50% test
split_id = int(split_frac * len(test_sentences))
val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

pickle.dump(train_sentences, open(f'dataset/train_sentences.pkl', 'wb'))
pickle.dump(val_sentences, open(f'dataset/val_sentences.pkl', 'wb'))
pickle.dump(test_sentences, open(f'dataset/test_sentences.pkl', 'wb'))
pickle.dump(train_labels, open(f'dataset/train_labels.pkl', 'wb'))
pickle.dump(val_labels, open(f'dataset/val_labels.pkl', 'wb'))
pickle.dump(test_labels, open(f'dataset/test_labels.pkl', 'wb'))

pickle.dump(word2idx, open(f'dataset/word2idx.pkl', 'wb'))
pickle.dump(idx2word, open(f'dataset/idx2word.pkl', 'wb'))
