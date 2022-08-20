import sys
import os
import numpy as np
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier

stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

DEBUG = True

"""
    kNN classifier
    Extracts tokens, computes feature matrices, and trains & tests the kNN classifier
"""


class kNN_classifier:
    def __init__(self, data_path, out_file):
        self.data_path = data_path
        self.out_file = out_file
        self.tokens = []
        self.feature_idx_map = {}
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    """ Maps features (tokens) to integers """

    def prepare_feature_map(self):
        for pos, token in enumerate(self.tokens):
            self.feature_idx_map[token] = pos

    """ Computes the feature matrix of size (n_samples, n_features) """

    def prepare_count_matrix(self):
        classes = os.listdir(self.data_path)
        for className in classes:
            class_path = os.path.join(self.data_path, className)
            if className == "class1" or className == "class2":
                classID = int(className[-1])
                data_folders = os.listdir(class_path)
                for data_folder in data_folders:
                    data_path = os.path.join(class_path, data_folder)
                    files = os.listdir(data_path)
                    files.sort(key=lambda x: int(x))
                    X = []
                    y = []
                    for file_ in files:
                        text = open(os.path.join(data_path, file_), errors="replace").read()
                        text = text.lower()
                        feature_vector = [0] * len(self.tokens)
                        for _ in punctuation:
                            text = text.replace(_, " ")
                        text = text.replace("  ", " ")
                        tokens = word_tokenize(text)
                        for token in tokens:
                            if token not in stopwords:
                                token = lemmatizer.lemmatize(token)
                                try:
                                    pos = self.feature_idx_map[token]
                                    feature_vector[pos] += 1
                                except KeyError:
                                    pass
                        X.append(feature_vector)
                        y.append(classID)

                    if data_folder == "train":
                        self.X_train.extend(X)
                        self.y_train.extend(y)
                    else:
                        self.X_test.extend(X)
                        self.y_test.extend(y)

        if DEBUG:
            print("Construction of feature matrix complete")
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

    """ Uses TfidfTransformer to compute the tf-idf matrices from the feature matrices """

    def prepare_tf_idf_vectors(self):
        transformer = TfidfTransformer(sublinear_tf=True, smooth_idf=False)
        self.X_train = transformer.fit_transform(self.X_train).toarray()
        self.X_test = transformer.transform(self.X_test).toarray()

    """ Reads the dataset folder"""

    def read_dataset(self):
        classes = os.listdir(self.data_path)
        for className in classes:
            class_path = os.path.join(self.data_path, className)
            if className == "class1" or className == "class2":
                self.tokens.extend(self.read_class(class_path))
        self.tokens = list(set(self.tokens))
        self.tokens.sort()
        if DEBUG:
            print("Total Features: {}".format(len(self.tokens)))

    """ Reads data files for each class (class1 and class2) """

    def read_class(self, class_path):
        data_folders = os.listdir(class_path)
        for data_folder in data_folders:
            data_path = os.path.join(class_path, data_folder)
            if data_folder == "train":
                return self.process_data(data_path)

    """ Computes tokens from the texts in all files pointed by 'data_path' """

    @staticmethod
    def process_data(data_path):
        files = os.listdir(data_path)
        features = []
        cache = {}
        files.sort(key=lambda x: int(x))
        for file_ in files:
            text = open(os.path.join(data_path, file_), errors="replace").read()
            text = text.lower()
            for _ in punctuation:
                text = text.replace(_, " ")
            text = text.replace("  ", " ")
            for token in word_tokenize(text):
                if token not in stopwords:
                    token = lemmatizer.lemmatize(token)
                    if token not in cache.keys():
                        features.append(token)
                        cache[token] = 1
        return features

    """ Computes Inner product of two vectors x and y """

    @staticmethod
    def inner_product(x, y):
        return 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    """ Runs the kNN classifier for different values of b """

    def run_kNN(self, K, out_file):

        results = []
        for k in K:
            kNN = KNeighborsClassifier(n_neighbors=k, metric=self.inner_product)
            kNN.fit(self.X_train, self.y_train)
            y_predict = kNN.predict(self.X_test)
            score = f1_score(self.y_test, y_predict, average="macro")
            results.append(score)
            if DEBUG:
                print("k = {}, f1_score = {}".format(k, score))

        result_file = open(out_file, "w", encoding="utf-8")
        result_file.write("k   ")

        for k in K:
            result_file.write("{}        ".format(k))
        result_file.write("\n")

        result_file.write("kNN")
        for result in results:
            result_file.write(" {0:.6f}".format(result))


if __name__ == "__main__":
    data_path, out_file = sys.argv[1], sys.argv[2]
    K = [1, 10, 50]
    kNN = kNN_classifier(data_path, out_file)
    kNN.read_dataset()
    kNN.prepare_feature_map()
    kNN.prepare_count_matrix()
    kNN.prepare_tf_idf_vectors()
    kNN.run_kNN(K, out_file)
