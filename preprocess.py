import numpy as np
import sys
from random import seed, shuffle
import csv

def read_file(file_name):
    """
    helper function to read a .csv file.

    :param file_name: string, name of data file
    :return: a 2d np array where the first index is a molecule
             and the second index is a string of comma separated
             scents.
    """
    # open the file passed in
    with open(file_name, newline='') as csvfile:
        # in read mode
        reader = csv.reader(csvfile, delimiter=' ')
        molecules = []
        labels = []
        for row in reader:
            # Split on only the first column, separating molecule and words
            strs = row[0].split(',', 1)
            # avoid garbage
            if len(strs) > 1:
                # replace redundant string quotes
                strs[1] = strs[1].replace('"', '')
                molecules.append(strs[0])
                labels.append(strs[1])
        # stack the arrays
        data = np.stack([molecules[1:], labels[1:]])
        # print(file_name, data)
        return data

def get_data(test_file, train_file, vocab_file, num_scents=3):
    """
    A function to get the data from CSV, files, and contruct

    The goal here is to construct a few numpy arrays

    :param train_file:  path to training csv
           test_file:   path to training csv
           vocab_file:  path to vocabulary txt file
    :return: train:     3d array of size [training examples, 1, num_scents]
             test:      3d array of size [testing examples, 1, num_scents]
             vocab      a dictionary mapping {index: word}
    """

    # Read in training and testing files
    train = read_file(train_file)
    test = read_file(test_file)

    # open the vocabulary file, read it in, and split on newlines
    f = open(vocab_file, "r")
    words = f.read()
    words = words.split('\n')

    # make a dictionary that maps word->int
    vocab = {words[i]:i for i in range(len(words))}

    # # construct labs, an array of size [emxamples, 3]
    # labs = []
    # # replace words with dictionary values
    # for i in range(len(train[1])):
    #     labels = train[1, i]
    #     scents = labels.split(",")
    #
    #     # replace with ints
    #     three_scents = []
    #     for j in range(num_scents):
    #         if len(scents) > j:
    #             # if there exists another scent for this molecule...
    #             three_scents.append(vocab[scents[j]])
    #         else:
    #             # when there are no more scents, append empty str
    #             three_scents.append(vocab[''])
    #     labs.append(three_scents)
    #
    # # cast to numpy array, then stack them into train
    # labs = np.array(labs)
    # train = np.stack([train[0], labs[:, 0], labs[:, 1], labs[:, 2]])
    # print(train)

    return train, test, vocab
if __name__ == "__main__":
    # read_file('./data/train.csv')
    get_data('./data/test.csv', './data/train.csv', './data/vocab.txt')
