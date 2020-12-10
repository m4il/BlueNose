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
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning

class Model(tf.keras.Model):
    """Model class representing your MPNN."""

    def __init__(self, vocab_size):
        """
        Instantiate a lifting layer, an optimizer, some number of MPLayers
        (we recommend 3), and a readout layer.
        """
        super(Model, self).__init__()

        # TODO: Initialize hyperparameters
        self.num_classes = vocab_size
        # self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=200, decay_rate=0.95, staircase=True)
        self.learning_rate = 3e-4
        self.hidden_size = 300
        self.batch_size = 100

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.liftLayer = tf.keras.layers.Dense(self.hidden_size)
        self.readoutLayer = tf.keras.layers.Dense(self.num_classes, activation='relu')
        self.D1 = tf.keras.layers.Dense(self.num_classes, activation='relu')
        self.Dropout = tf.keras.layers.Dropout(0.2)
        self.D2 = tf.keras.layers.Dense(self.num_classes, activation='relu')
        self.Dropout2 = tf.keras.layers.Dropout(0.1)
        self.D3 = tf.keras.layers.Dense(self.num_classes)
        self.Dropout3 = tf.keras.layers.Dropout(0.75)

        self.mp = MPLayer(self.hidden_size,self.hidden_size)
        self.mp1 = MPLayer(self.hidden_size,self.hidden_size)
        self.mp2 = MPLayer(self.hidden_size,self.hidden_size)

    def call(self, g):
        """
        Computes the forward pass of your network.
        1) Lift the features of the batched graph passed in. Don't apply an activation function.
        2) After the node features for the graph have been lifted, run them
           through the MPLayers.
        3) Feed the output of the final MPLayer through the readout function
           to get the logits.
        :param g: The DGL graph you wish to run inference on.
        :return: logits tensor of size (batch_size, 2)
        """
        g.ndata['node_feats'] = self.liftLayer(g.ndata['node_feats'])
        self.mp.call(g)
        self.mp1.call(g)
        self.mp2.call(g)
        read = self.readout(g,g.ndata['node_feats'])
        d_1 = self.D1(read)
        drop1 = self.Dropout2(d_1)
        d_2 = self.D2(drop1)
        drop2 = self.Dropout2(d_2)
        d_3 = self.D3(drop2)
        logits = self.Dropout3(d_3)
        return logits # check whether loss function requires set/list..

    def readout(self, g, node_feats):
        """
        Reduces the dimensionality of the graph to
        num_classes, and sums the node features in order to return logits.
        :param g: The batched DGL graph
        :param node_feats: The features at each node in the graph. Tensor of shape
                                   (num_atoms_in_batched_graph,
                                    size_of_node_vectors_from_prev_message_passing)
        :return: logits tensor of size (batch_size, vocab_size)
        """
        # TODO: Set the node features to be the output of your readout layer on
        # node_feats, then use dgl.sum_nodes to return logits.
        g.ndata['node_feats'] = self.readoutLayer(node_feats)

        return dgl.sum_nodes(g,'node_feats')


    def recall_function(self, logits, labels):
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

        # vals, idxs = tf.math.top_k(prbs, 5) #arbitrarily snag the top 5
        # preds = np.zeros((self.batch_size, self.num_classes))
        # for i in range(len(idxs)): # set each of the 5 options for each row = 1
        #     preds[i, idxs[i]] = 1
        # intersection = tf.math.reduce_sum(tf.math.multiply(labels, preds))
        # sum_ = tf.math.reduce_sum(labels + preds)
        # jac = (intersection + smooth) / (sum_ - intersection + smooth)
        # # for pred, label in zip(prbs, labels):
        # #     vals, idxs = tf.math.top_k(pred, tf.cast(tf.math.count_nonzero(label), tf.int32))
        # #     vals_l, idxs_l = tf.math.top_k(label, tf.cast(tf.math.count_nonzero(label), tf.int32))
        # #     print("correct", idxs, idxs_l)
        # ret = (1-jac)*smooth
        # print(ret)


        # intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
        # sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred), axis=-1)
        # jac = (intersection + smooth) / (sum_ - intersection + smooth)
        # return (1 - jac) * smooth


class MPLayer(Layer):
    """
    A TensorFlow Layer designed to represent a single round of message passing.
    This should be instantiated in your Model class several times.
    """

    def __init__(self, in_feats, out_feats):
        """
        Make a message computation layer which will compute the messages sent
        by each node to its neighbors and an output layer which will be
        applied to all nodes as a final transformation after message passing
        from size in_feats to out_feats.
        :param in_feats: The size of vectors at each node of your graph when you begin
        message passing for this round.
        :param out_feats: The size of vectors that you'd like to have at each of your
        nodes when you end message passing for this round.
        """
        super(MPLayer, self).__init__()

        self.num_atoms = 119
        self.num_bonds = 5

        self.bondLayer = tf.keras.layers.Dense(1, activation='relu')
        self.messageLayer = tf.keras.layers.Dense(in_feats, activation = 'relu')
        self.outputLayer = tf.keras.layers.Dense(out_feats, activation = 'relu')


    def call(self, g):
        """
        Computes the forward pass of your MPNN layer
        1) Call the either DGL's send and receive function or your own,
            depending on the is_testing flag
        2) Calculate the output features by passing the graph's node features
            through the output layer
        3) Set the graph's node features to be the output features

        To assign/retrieve the node data, you can use `g.ndata["node_feats"]`

        The send and receive functions to be used are the following:
            g.send_and_recv                 # DGL send_and_recv
            custom_send_and_recv            # custom send_and_recv
        We assign the "messager" function and the "reducer" function to be
            passed in to the send and receive function for you

        :param g: The batched DGL graph you wish to run inference on.
        :param is_testing: True if using custom send_and_recv, false if using DGL
        :return: None
        """
        # The message function for testing
        messager = lambda x: self.message(x)
        # The reduce function for testing
        reducer = lambda x: self.reduce(x)
        # TODO: Fill this in!

        g.send_and_recv(g.edges(),messager,reducer)
        g.ndata['node_feats'] = self.outputLayer(g.ndata['node_feats'])

        return None


    def message(self, edges):
        """
        This function, when called on a group of edges, should compute a message
        for each edge to be sent from the edge's src node to its dst node.

        The message is computed by passing the src node's features into a linear layer
        with ReLU activation. This message will be sent to all dst nodes at once
        by sending a dictionary with key 'msg' to a shared mailbox.

        The source nodes' features can all be accessed like:
            edges.src['node_feats']    # DGL send_and_recv
            edges                      # custom send_and_recv

        :param edges: All the DGL edges in the batched DGL graph.
        :param is_testing: True if using custom send_and_recv, false if using DGL
        :return: A dictionary from some 'msg' to all the messages
        computed for each edge.
        """

        mul = tf.math.multiply(tf.transpose(edges.src['node_feats']), edges.data['edge_feats'])
        out = tf.transpose(mul)
        msg = self.messageLayer(out)
        # msg = self.messageLayer(edges.src['node_feats'])
        return {'msg' : msg}

    def reduce(self, nodes, is_testing=False):
        """
        This function, when called on a group of nodes, should aggregate (i.e. sum)
        the messages in the mailboxes of each node. Each node will only have messages
        from its neighbors.

        We will then save these new features in each node under the attribute 'node_feats'.
        The messages of all nodes can be accessed like:
            nodes.mailbox['msg']    # DGL send_and_recv
            nodes['msg']            # custom send_and_recv

        :param nodes: All the DGL nodes in the batched DGL Graph.
        :param is_testing: True if using custom send_and_recv, false if using DGL
        :return: A dictionary from 'node_feats' to the summed messages for each node.
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
    Trains your model given the training data.
    For each batch of molecules in train data...
        1) Make dgl graphs for each of the molecules in your batch; collect them in a list.
        2) Call dgl.batch to turn your list of graphs into a batched graph.
        3) Turn the labels of each of the molecules in your batch into a 1-D tensor of size
            batch_size
        4) Pass this graph to the Model's forward pass. Run the resulting logits
                        and the labels of the molecule batch through SparseCategoricalCrossentropy.
        5) Compute the gradients of the Model's trainable variables.
        6) Take a step with the optimizer.
    :param model: Model class representing your MPNN.
    :param train_data: A 1-D list of molecule objects, representing all the molecules
    in the training set from get_data
    :return: nothing.
    """
    # This is the loss function, usage: loss(labels, logits)
    # TODO: Implement train with the docstring instructions
    max_divisor = len(train_data) - (len(train_data) % model.batch_size)
    losses = []

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
            # loss = model.loss_function(logits, labels)
            loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))
        # print(model.trainable_variables)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.opt.apply_gradients(zip(gradients, model.trainable_variables))
        loss = tf.math.reduce_mean(loss)
        losses.append(loss)
        # print(model.opt._decayed_lr(tf.float32))
        # if k % 50 == 0:
        #     print(loss)

    return losses

def test(model, test_data, test_labels):
    """
    Testing function for our model.
    Batch the molecules in test_data, feed them into your model as described in train.
    After you have the logits: turn them back into numpy arrays, compare the accuracy to the labels,
    and keep a running sum.
    :param model: Model class representing your MPNN.
    :param test_data: A 1-D list of molecule objects, representing all the molecules in your
    testing set from get_data.
    :return: total accuracy over the test set (between 0 and 1)
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
    # TODO: Return the training and testing data from get_data
    # TODO: Instantiate model
    # TODO: Train and test for up to 15 epochs.
    train_mols, train_labs, valid_mols, valid_labs, test_mols, vocab = get_data('./data/test.csv', './data/train.csv', \
     './data/vocab.txt')

    t_loss = []
    t_acc = []
    t_rec = []
    m = Model(len(vocab))
    for i in range(75):
        print("training...", i)
        loss = train(m, train_mols, tf.convert_to_tensor(train_labs, dtype=tf.float32))
        t_loss = t_loss + list(loss)
        print("testing...", i)
        acc, rec = test(m, valid_mols, tf.convert_to_tensor(valid_labs, dtype=tf.float32))
        t_acc.append(acc)
        t_rec.append(rec)
        print("testing accuracy:", acc, rec)
    visualize_loss(t_loss)
    visualize_accuracy(t_acc, t_rec)


if __name__ == '__main__':
    main()
