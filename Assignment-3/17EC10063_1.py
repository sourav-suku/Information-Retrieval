import sys
import os
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import numpy as np
from string import punctuation
from sklearn.metrics import f1_score

stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

DEBUG = True

"""
    Naive Bayes classifier
    Extracts tokens, computes feature matrices and trains & tests the Multinomial and Bernoulli NB classifiers
"""


class Naive_Bayes:
    def __init__(self, data_path, out_file):
        self.data_path = data_path
        self.out_file = out_file
        self.tokens = []
        self.feature_idx_map = {}
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    """ Computes the feature matrix of size (n_samples, n_features) """

    def generate_feature_matrix(self):
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
                        for token in word_tokenize(text):
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

    """ Maps features (tokens) to integers """

    def create_feature_map(self):
        for pos, token in enumerate(self.tokens):
            self.feature_idx_map[token] = pos

    """ Reads the dataset folder"""

    def read_dataset(self):
        classes = os.listdir(self.data_path)
        for className in classes:
            class_path = os.path.join(self.data_path, className)
            if className == "class1":
                self.tokens.extend(self.read_class(class_path))
            if className == "class2":
                self.tokens.extend(self.read_class(class_path))
        self.tokens = list(set(self.tokens))
        self.tokens.sort()
        if DEBUG:
            print("Total Features: {}".format(len(self.tokens)))

    """ Reads data files for each class (class1 and class2)"""

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

    """ Fits MultinomialNB on X_train and predicts on X_test """

    def fit_MultinomialNB(self, X_train, X_test):
        multinomialNB = MultinomialNB()
        multinomialNB.fit(X_train, self.y_train)
        y_predict = multinomialNB.predict(X_test)
        score = f1_score(self.y_test, y_predict, average="macro")
        return score

    """ Fits BernoulliNB on X_train and predicts on X_test """

    def fit_BernoulliNB(self, X_train, X_test):
        bernoulliNB = BernoulliNB()
        bernoulliNB.fit(X_train, self.y_train)
        y_predict = bernoulliNB.predict(X_test)
        score = f1_score(self.y_test, y_predict, average="macro")
        return score

    """ Runs both NB classifiers on different values of top features """

    def run_NB(self, out_file, top_features_count):

        multinomialNB_scores = []
        bernoulliNB_scores = []

        for count in top_features_count:
            if DEBUG:
                print("Computing results for x = {}".format(count))
            top_features = SelectKBest(mutual_info_classif, k=count)
            X_train = top_features.fit_transform(self.X_train, self.y_train)
            X_test = top_features.transform(self.X_test)

            multinomialNB_score = self.fit_MultinomialNB(X_train, X_test)
            bernoulliNB_score = self.fit_BernoulliNB(X_train, X_test)

            multinomialNB_scores.append(multinomialNB_score)
            bernoulliNB_scores.append(bernoulliNB_score)

        self.write_answer(out_file, top_features_count, multinomialNB_scores, bernoulliNB_scores)

    """ Writes answer to the output file """

    def write_answer(self, out_file, top_features_count, multinomialNB_scores, bernoulliNB_scores):

        result_file = open(out_file, "w", encoding="utf-8")

        result_file.write("NumFeature    ")

        for count in top_features_count:
            result_file.write("{}        ".format(count))

        result_file.write("\nMultinomialNB")

        for pos in range(len(top_features_count)):
            result_file.write(" {0:.6f}".format(multinomialNB_scores[pos]))

        result_file.write("\nBernoulliNB  ")

        for pos in range(len(top_features_count)):
            result_file.write(" {0:.6f}".format(bernoulliNB_scores[pos]))

        result_file.close()


if __name__ == "__main__":
    data_path, out_file = sys.argv[1], sys.argv[2]
    top_features_count = [1, 10, 100, 1000, 10000]
    NB = Naive_Bayes(data_path, out_file)
    NB.read_dataset()
    NB.create_feature_map()
    NB.generate_feature_matrix()
    NB.run_NB(out_file, top_features_count)
