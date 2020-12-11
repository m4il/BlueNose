from preprocess import get_data
from tensorflow.keras.layers import Dense, Layer
from periodictable import elements
import tensorflow as tf
import numpy as np
import os
import pysmiles
import networkx
import itertools
os.environ['DGLBACKEND'] = "tensorflow"
import dgl
import numpy as np
import matplotlib.pyplot as plt
import sys

import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # mute pysmiles print statements

class Model(tf.keras.Model):
    """Model class, aka GNN."""

    def __init__(self, vocab_size):
        super(Model, self).__init__()

        # hyper parameters
        self.num_classes = vocab_size
        self.learning_rate = 3e-4
        self.hidden_size = 300
        self.batch_size = 100
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        #lifing layer, 3 dense layers with dropout
        self.liftLayer = tf.keras.layers.Dense(self.hidden_size)
        self.readoutLayer = tf.keras.layers.Dense(self.num_classes, activation='relu')
        self.D1 = tf.keras.layers.Dense(self.num_classes, activation='relu')
        self.Dropout = tf.keras.layers.Dropout(0.2)
        self.D2 = tf.keras.layers.Dense(self.num_classes, activation='relu')
        self.Dropout2 = tf.keras.layers.Dropout(0.1)
        self.D3 = tf.keras.layers.Dense(self.num_classes)
        self.Dropout3 = tf.keras.layers.Dropout(0.75)

        # 3 graph message passes
        self.mp = MPLayer(self.hidden_size,self.hidden_size)
        self.mp1 = MPLayer(self.hidden_size,self.hidden_size)
        self.mp2 = MPLayer(self.hidden_size,self.hidden_size)

    def call(self, g):
        """
        Computes a forward pass on the network.

        :param g: DGL graph.
        :return: logits tensor of size (batch_size, vocab_size)
        """
        # lift node feats, 3 graph message passes
        g.ndata['node_feats'] = self.liftLayer(g.ndata['node_feats'])
        self.mp.call(g)
        self.mp1.call(g)
        self.mp2.call(g)
        read = self.readout(g,g.ndata['node_feats'])
        # apply dense layers
        d_1 = self.D1(read)
        drop1 = self.Dropout2(d_1)
        d_2 = self.D2(drop1)
        drop2 = self.Dropout2(d_2)
        d_3 = self.D3(drop2)
        logits = self.Dropout3(d_3)
        return logits

    def readout(self, g, node_feats):
        """
        Reduce dimensionality of graph summing over the mailboxes.

        :param g: DGL graph
        :param node_feats: node features, one hot with other data appended
        :return: logits, size (batch_size, vocab_size)
        """

        g.ndata['node_feats'] = self.readoutLayer(node_feats)
        return dgl.sum_nodes(g,'node_feats')


    def recall_function(self, logits, labels):
        """
        Home made accuracy function
        """
        t_rec = 0
        for pred, label in zip(logits, labels):
            rec = 0
            idx = tf.math.top_k(pred, tf.cast(tf.math.count_nonzero(label), tf.int32))[1]
            for i in idx:
                if label[i] == 1:
                    rec = 1
            t_rec += rec
        return t_rec/self.batch_size

    def accuracy_function(self, y_pred, y_true):
        """
        inspired by:

            Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                    = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

            The jaccard distance loss is usefull for unbalanced datasets. This has been
            shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
            gradient.

            Ref: https://en.wikipedia.org/wiki/Jaccard_index

            @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
            @author: wassname
        """
        acc = 0
        for pred, label in zip(y_pred, y_true):
            intersection = 0
            union = tf.math.count_nonzero(label)*2
            idx = tf.math.top_k(pred, tf.cast(tf.math.count_nonzero(label), tf.int32))[1]
            for i in idx:
                if label[i] == 1:
                    intersection += 1
            union -= intersection
            acc += intersection/union
        acc /= self.batch_size
        return acc


    def loss_function(self, logits, labels, smooth=100):
        prbs = tf.math.sigmoid(logits)
        for pred, label in zip(prbs, labels):
            intersection = 0
            sum_ = 0
            vals, idxs = tf.math.top_k(pred, tf.cast(tf.math.count_nonzero(label), tf.int32))
            for i,v in zip(idxs, vals):
                sum_ += (label[i] + tf.math.abs(v))
                intersection += tf.math.abs(label[i] * v)
            jac = (intersection + smooth) / (sum_ - intersection + smooth)
            res = (1-jac)*smooth
        return res


class MPLayer(Layer):
    """
    Layer to represent messsage passing.
    """

    def __init__(self, in_feats, out_feats):
        """
        :param in_feats: dimensionality of the in features
        :param out_feats: dimensionality of the out features
        """
        super(MPLayer, self).__init__()

        #instatniate dense layers
        self.bondLayer = tf.keras.layers.Dense(1, activation='relu')
        self.messageLayer = tf.keras.layers.Dense(in_feats, activation = 'relu')
        self.outputLayer = tf.keras.layers.Dense(out_feats, activation = 'relu')


    def call(self, g):
        """
        Computes one message pass

        :param g: The batched DGL graph.
        """

        messager = lambda x: self.message(x)
        reducer = lambda x: self.reduce(x)

        g.send_and_recv(g.edges(),messager,reducer)
        g.ndata['node_feats'] = self.outputLayer(g.ndata['node_feats'])

        return None


    def message(self, edges):
        """
        Multiply node features by bond strength, then pass through dense
        messageLayer.

        :param edges: graph edges
        :return: A dictionary mapping 'msg': messages
        """

        # must transpose so tensorflow broadcasts
        mul = tf.math.multiply(tf.transpose(edges.src['node_feats']), edges.data['edge_feats'])
        out = tf.transpose(mul)
        msg = self.messageLayer(out)
        return {'msg' : msg}

    def reduce(self, nodes, is_testing=False):
        """
        Aggregates messages, sums them, then places them back into node_feats

        :param nodes: nodes in the graph
        :return: dictionary mapping 'node_feats': summed messages
        """
        return {'node_feats' : tf.math.reduce_sum(nodes.mailbox['msg'], axis=1)}


def pt_lookup(atoms):
    """
    Takes in a tensor of atoms in byte strings, and converts them to one hot
    tensors corresponding to their atomic number.

    :param atoms: a tensor of byte strings
    :return: one hot vector of shape (num atoms, 119) corresponding to atomic numbers
    """
    periodic_table = {} # Need atomic number as feature instead of symbol.
    for el in elements:
        periodic_table[el.symbol] = el.number
    elems = []
    for atom in atoms.numpy():
        # atom comes in as a byte string, so we decode
        symbol = atom.decode("utf-8")
        elems.append(periodic_table[str(atom)[2:-1]])

    # convert to one hot
    e_hot = tf.one_hot(elems, len(periodic_table)+2)
    return e_hot

def build_graph(smiles):
    """
    Constructs a NetworkX graph out of a SMILES representation of a molecule from the train/test data.
    :param smiles: a string object of SMILES format
    :return: nx.Graph:
        A graph describing a molecule. Nodes will have an 'element', 'aromatic'
        and a 'charge', and if `explicit_hydrogen` is False a 'hcount'.
        Depending on the input, they will also have 'isotope' and 'class'
        information.
        Edges will have an 'order'.
    """

    '''
    can access node data and edge data when the graph is in networkx format
    dgl.from_networkx(g) converts networkx to dgl graph but the node data and edge data doesnt seem to be transferred
    Goal: save the node feats and edge feats of networkx as tensor and set them to dgl graph ndata and edata
    Question: Do we save ndata as ('C', 'C', 'C', 'O', 'C') or do we create one hot vectors like in the hw
    '''
    # read the smile graphs in using pysmiles & build network
    g = pysmiles.read_smiles(smiles)

    # get the features from the graph and convert to tensor
    elems = g.nodes(data='element')
    h_count = g.nodes(data='hcount')
    aros = g.nodes(data='aromatic')
    raw_node_feats = []
    for elem, data, aro in zip(elems, h_count, aros):
        node = list(elem)
        node.append(data[1])
        node.append(aro[1]*1)
        raw_node_feats.append(node)
    na = np.array(list(raw_node_feats))
    byte_node_feats = tf.convert_to_tensor(na[:,1])

    # turn the byte string node feats into one_hot node feats
    node_feats = pt_lookup(byte_node_feats).numpy()
    node_feats[:, -2] = na[:, 2]
    node_feats[:, -1] = na[:, 3]
    node_feats = tf.convert_to_tensor(node_feats)

    # get edge data and extract bonds, double them, then convert to tensor
    edata = g.edges(data='order')
    bonds = list(edata)
    na = np.array(bonds)
    tup = zip(na[:,2], na[:,2])
    bond_data = tf.convert_to_tensor(list(itertools.chain(*tup)))
    bond_data = tf.cast(bond_data, tf.float32)
    # build dgl graph
    dgl_graph = dgl.from_networkx(g)

    dgl_graph.ndata['node_feats'] = node_feats
    dgl_graph.edata['edge_feats'] = bond_data

    return dgl_graph


def train(model, train_data, train_labels):
    """
    Train the model for one epoch
    :param model: model class
    :param train_data: 1d tensor of smiles
    :param train_labels: 1d tensor of multi_hots
    """

    max_divisor = len(train_data) - (len(train_data) % model.batch_size)
    losses = []
    ##################### shuffle #########################
    # idxs = np.arange(len(train_data))
    # np.random.shuffle(idxs)
    # tf.convert_to_tensor(idxs)
    # train_data = train_data[idxs]
    # tf.gather(train_labels, idxs)

    for k in range(0, max_divisor, model.batch_size):
        batch_inputs = train_data[k:(k+model.batch_size)]
        labels = tf.convert_to_tensor(train_labels[k:(k+model.batch_size)])
        graphs = []

        for m in batch_inputs:
            graphs.append(build_graph(m))

        g_batched = dgl.batch(graphs)
        with tf.GradientTape() as tape:
            logits = model.call(g_batched)
            # loss = model.loss_function(logits, labels) # use this to run custom loss func
            loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.opt.apply_gradients(zip(gradients, model.trainable_variables))
        loss = tf.math.reduce_mean(loss)
        losses.append(loss)

    return losses

def test(model, test_data, test_labels):
    """
    Test out model on the inputs

    :param model: model class for GNN
    :param train_data: 1d tensor of smiles
    :param train_labels: 1d tensor of multi_hots
    :return: list of accuracies and recalls, all between (0,1)
    """

    max_divisor = len(test_data) - (len(test_data) % model.batch_size)
    acc = []
    rec = []
    for k in range(0, max_divisor, model.batch_size):
        batch_inputs = test_data[k:(k+model.batch_size)]
        labels = tf.convert_to_tensor(test_labels[k:(k+model.batch_size)])
        graphs = []

        for m in batch_inputs:
            graphs.append(build_graph(m))

        g_batched = dgl.batch(graphs)
        logits = model.call(g_batched)
        acc.append(model.accuracy_function(logits, labels))
        rec.append(model.recall_function(logits, labels))

    return tf.math.reduce_mean(acc), tf.math.reduce_mean(rec)

def visualize_loss(loss):
    x = np.arange(0, len(loss))
    y = loss
    plt.title("Losses")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(x, y, color ="red")
    plt.show()

def visualize_accuracy(acc, rec):
    x = np.arange(0, len(acc))
    y = acc
    a = np.arange(0, len(rec))
    b = rec
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage")
    plt.plot(x, y, color ="blue")
    plt.plot(a, b, color = "red")
    plt.show()


def main():
    train_mols, train_labs, valid_mols, valid_labs, test_mols, vocab = get_data('./data/test.csv', './data/train.csv', \
     './data/vocab.txt')

    t_loss = []
    t_acc = []
    t_rec = []
    m = Model(len(vocab))
    for i in range(5):
        # print("training...", i)
        loss = train(m, train_mols, tf.convert_to_tensor(train_labs, dtype=tf.float32))
        t_loss = t_loss + list(loss)
        # print("testing...", i)
        acc, rec = test(m, valid_mols, tf.convert_to_tensor(valid_labs, dtype=tf.float32))
        t_acc.append(acc)
        t_rec.append(rec)
        print("testing accuracy:", acc, rec)
    # visualize_loss(t_loss)
    # visualize_accuracy(t_acc, t_rec)


if __name__ == '__main__':
    main()
