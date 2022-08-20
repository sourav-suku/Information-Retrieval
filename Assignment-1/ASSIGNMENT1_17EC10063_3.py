from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import string
import json
import time


DEBUG = False
DATA_PATH = './ECTText/'

inverted_index = dict()

lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

''' Sorting function to sort input files in lexicographically increasing order '''


def sortKey(s):
    return int(s.split('.')[0])


files = os.listdir(DATA_PATH)
files.sort(key=sortKey)

''' Extract the tokens after stopwords | punctuation removal followed by lemmatization
    and build the inverted_index '''


def build_inverted_index():

    file_num = 0
    for file in files:
        with open(os.path.join(DATA_PATH, file), 'r', encoding='utf-8') as ECTText:
            text = ECTText.read().replace('\n', ' ').lower().strip()
            position = 0
            for token in word_tokenize(text):
                # Remove stop words & punctuation marks
                if token not in stopwords and token not in string.punctuation:
                    lemma = lemmatizer.lemmatize(token)
                    try:
                        inverted_index[lemma].append((file_num, position))
                    except KeyError:
                        inverted_index[lemma] = [(file_num, position)]
                    position = position + 1
            file_num += 1
            if DEBUG and file_num % 100 == 0:
                print("Tokenization - Steps done: {} | Tokens found: {}".format(
                    file_num, len(inverted_index.keys())))

    if DEBUG:
        print("Total number of tokens: {}".format(len(inverted_index.keys())))
    with open('inverted_index.json', 'w') as inv_idx:
        json.dump(inverted_index, inv_idx)


if __name__ == "__main__":
    start_time = time.time()
    build_inverted_index()
    if DEBUG:
        print("--- %s seconds ---" % (time.time() - start_time))
