import numpy as np
import sys
import csv

def read_file(file_name):
    """
    helper function to read a .csv file.

    :param file_name: string, name of data file
    :return: molecules: np array of SMILE molecules in string format
             labels:    np array of comma-separated labels
    """
    # open the file passed in
    with open(file_name, newline='') as csvfile:
        # in read mode
        reader = csv.reader(csvfile, delimiter=' ')
        molecules = []
        labels = []
        i=0
        for row in reader:
            i+=1
            # Split on only the first column, separating molecule and words
            strs = row[0].split(',', 1)
            # avoid garbage
            if strs != ['']:
                molecules.append(strs[0])
                # deal with labels
                if len(strs) > 1:
                    # replace redundant string quotes
                    strs[1] = strs[1].replace('"', '')
                    labels.append(strs[1])

        mols = np.array(molecules[1:])
        labs = np.array(labels[1:])

        return mols, labs

def encode(labels, vocab):
    """
    A helper function used to convert a 2D data array into a 1d example
    array and a second 2D multi-hot array.

    :param labels: 1D array of size (examples,)
           vocab:  dictionary mapping {word: index}

    :return: data:      np array of examples size (num examples,)
             labels:    2D np array of labels size (num examples, vocab_size)
    """

    # for each scent, we are going to replace word with a key
    idx_labels = []
    # convert the labels to arrays, replacing words with indices
    for l in labels:
        # for each label, ie: ['lavendar,floral,earthy']
        indices = []
        scents = l.split(',') # split string
        for s in scents:
            # get the index of each individual scent
            indices.append(vocab[s])
        idx_labels.append(indices)
    idx_labels = np.array(idx_labels)

    # construct an array of 0s to use as multi-hot vectors
    z = np.zeros((len(labels), len(vocab)))
    for i in range(len(idx_labels)):
        for l in range(len(idx_labels[i])):
            z[i, idx_labels[i][l]] = 1
    return z

def get_data(test_file, train_file, vocab_file, num_scents=3):
    """
    A function to get the data from CSV, files, and contruct

    The goal here is to construct a few numpy arrays

    Training examples: 4135
    Testing examples: 1079
    Vocabulary size: 110

    :param train_file:  path to training csv
           test_file:   path to training csv
           vocab_file:  path to vocabulary txt file

    :return: train_mol:     np array of size (training examples*0.9,)
             train_labels:  np array of lists, size (training examples*0.9,)
             valid_mol:     np array of size (testing examples*0.1,)
             valid_labels:  np array of lists, size (training examples*0.1,)
             test_data:     np array of size (testing examples,)
             vocab          a dictionary mapping {word: index}
    """

    # Read in training and testing files
    train_molecules, train_labels = read_file(train_file)
    test_molecules, _ = read_file(test_file)

    # open the vocabulary file, read it in, and split on newlines
    f = open(vocab_file, "r")
    words = f.read()
    words = words.split('\n')

    # make a dictionary that maps word->int
    vocab = {words[i]:i for i in range(len(words))}

    # get multi hot vectors from the testing data
    train_labels = encode(train_labels, vocab)
    # print(train_labels)


    # split into training and validation sets
    split_idx = round(len(train_labels) * 0.9)

    train_molecules = train_molecules[:split_idx]
    valid_molecules = train_molecules[split_idx:]

    train_labels = train_labels[:split_idx]
    valid_labels = train_labels[split_idx:]
    # print(np.shape(train_labels))
    # print(np.shape(train_molecules), np.shape(train_labels), np.shape(test_molecules))
    return train_molecules, train_labels, valid_molecules, valid_labels, test_molecules, vocab

if __name__ == "__main__":
    get_data('./data/test.csv', './data/train.csv', './data/vocab.txt')
