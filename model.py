from preprocess import get_data
from tensorflow.keras.layers import Dense, Layer
from periodictable import elements
import tensorflow as tf
import numpy as np
import os
import pysmiles
import networkx
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
        self.raw_features = 119
        self.num_classes = vocab_size
        self.learning_rate = 3e-4
        self.hidden_size = 300
        self.batch_size = 10

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.liftLayer = tf.keras.layers.Dense(self.hidden_size)
        # self.readoutLayer = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')
        self.readoutLayer = tf.keras.layers.Dense(self.num_classes)

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
        logits = self.readout(g,g.ndata['node_feats'])
        # top_3 = tf.math.top_k(logits, k=3)
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

    # def accuracy_function(self, logits, labels):
    #     """
    #     Computes the accuracy across a batch of logits and labels.
    #     :param logits: a 2-D np array of size (batch_size, 2)
    #     :param labels: a 1-D np array of size (batch_size)
    #         (1 for if the molecule is active against cancer, else 0).
    #     :return: mean accuracy over batch.
    #     """
    #
    #     total = tf.math.reduce_sum(labels)
    #
    #     # get the indices of the top 3 predictions
    #     idx_preds = tf.math.top_k(logits)[1]
    #     idx_labels = tf.math.top_k(labels)[1]
    #
    #     # compare correct
    #     correct = tf.math.equal(idx_preds, idx_labels)
    #     # count correct
    #     total_correct = tf.cast(tf.math.count_nonzero(correct), tf.float32)
    #
    #     print(total_correct)
    #
    #     return tf.math.divide(total_correct, total)

    def accuracy_function(self, y_pred, y_true, smooth=100):
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
        acc /= len(y_pred)
        return acc


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
        # one layer for node feature weights
        # one later for edge feature weights
        # messageLayer combination of these two layers

        # srcs = edges.src['atomic_number']
        # dsts = edges.dst['atomic_number']
        #
        # e_idxs = tf.argmax(edges.data['edge_feats'], axis=1)
        # s_idxs = tf.argmax(srcs, axis=1)
        # d_idxs = tf.argmax(dsts, axis=1)
        #
        # # size: (num nodes, 114, 114, 5)
        # multi_hots = np.zeros((len(edges), self.num_atoms, self.num_atoms, self.num_bonds))
        #
        # c = 0
        # # find a way to vectorize this
        # for s,d,b in zip(s_idxs, d_idxs, e_idxs):
        #     multi_hots[c, s, d, b] = 1
        #     c += 1
        #
        # # this verifies we have it correct, we end up with 28 total nonzero indices,
        # # which is equal to the number of nodes
        # # print(np.count_nonzero(multi_hots), np.nonzero(multi_hots))
        #
        # res = self.bondLayer(tf.reshape(multi_hots, (len(edges), self.num_bonds*self.num_atoms*self.num_atoms)))
        #
        # # broadcast the num_edges x 1 bond representation to representation of node features
        # out = tf.math.multiply(res, self.messageLayer(edges.src['node_feats']))
        out = self.messageLayer(edges.src['node_feats'])
        return {'msg' : out}

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
    e_hot = tf.one_hot(elems, len(periodic_table))
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
    raw_node_feats = g.nodes(data='element')
    na = np.array(list(raw_node_feats))
    byte_node_feats = tf.convert_to_tensor(na[:,1])
    # turn the byte string node feats into one_hot node feats
    node_feats = pt_lookup(byte_node_feats)
    # print("node_feats",node_feats)

    # get edge data and extract bonds, then convert to tensor
    edata = g.edges(data='order')
    bonds = list(edata)
    na = np.array(bonds)
    t2 = tf.convert_to_tensor(na[:,2])

    # build dgl graph
    dgl_graph = dgl.from_networkx(g)

    # some fancy magic to set edata to the strength of bond
    edge_data = []
    dict = {frozenset((e1, e2)) : d for e1, e2, d in na}
    src, dest  = dgl_graph.edges()
    for e in zip(src.numpy(), dest.numpy()):
        bond = dict[frozenset(e)]
        if bond == 1.5:
            bond = 5
        edge_data.append(int(bond))
    edge_d_tensor = tf.convert_to_tensor(edge_data)
    #convert to one_hot tensor

    edge_oh = tf.one_hot(edge_d_tensor, 5)

    dgl_graph.ndata['node_feats'] = node_feats
    dgl_graph.ndata['atomic_number'] = node_feats
    dgl_graph.edata['edge_feats'] = edge_oh

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
    l = []
    for k in range(0, max_divisor, model.batch_size):
        batch_inputs = train_data[k:(k+model.batch_size)]
        labels = tf.convert_to_tensor(train_labels[k:(k+model.batch_size)])
        graphs = []

        for m in batch_inputs:
            graphs.append(build_graph(m))

        g_batched = dgl.batch(graphs)
        with tf.GradientTape() as tape:
            logits = model.call(g_batched)
            # print(logits)
            # logits = tf.dtypes.cast(logits[0], tf.int32)
            # losses = model.loss_function(logits, labels)
            losses = tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))
        l.append(losses)
        # print(model.trainable_variables)
        gradients = tape.gradient(losses, model.trainable_variables)
        model.opt.apply_gradients(zip(gradients, model.trainable_variables))

    return l

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
    for k in range(0, max_divisor, model.batch_size):
        batch_inputs = test_data[k:(k+model.batch_size)]
        labels = tf.convert_to_tensor(test_labels[k:(k+model.batch_size)])
        graphs = []

        for m in batch_inputs:
            graphs.append(build_graph(m))

        g_batched = dgl.batch(graphs)
        logits = model.call(g_batched)
        acc.append(model.accuracy_function(logits, labels))

    return tf.math.reduce_mean(acc)

def visualize_loss(loss):
    x = np.arange(0, len(loss))
    y = loss
    plt.title("Losses")
    plt.xlabel("Example")
    plt.ylabel("Loss")
    plt.plot(x, y, color ="red")
    plt.show()

def main():
    # TODO: Return the training and testing data from get_data
    # TODO: Instantiate model
    # TODO: Train and test for up to 15 epochs.
    train_mols, train_labs, valid_mols, valid_labs, test_mols, vocab = get_data('./data/test.csv', './data/train.csv', \
     './data/vocab.txt')

    # train_m = []
    # train_l = []
    # for mol, lab in zip(train_mols, train_labs):
    #     if tf.math.reduce_sum(lab) == 3:
    #         train_m.append(mol)
    #         train_l.append(lab)
    #
    # test_m = []
    # test_l = []
    # for mol, lab in zip(valid_mols, valid_labs):
    #     if tf.math.reduce_sum(lab) == 3:
    #         acc2 += 1
    #         test_m.append(mol)
    #         test_l.append(lab)
    t_loss = []
    m = Model(len(vocab))
    for i in range(1):
        print("training...", i)
        loss = train(m, train_mols, tf.convert_to_tensor(train_labs, dtype=tf.float32))
        t_loss = t_loss + list(loss)
        print("testing...", i)
        acc = test(m, valid_mols, tf.convert_to_tensor(valid_labs, dtype=tf.float32))
        print("testing accuracy:", acc)
    visualize_loss(loss)


if __name__ == '__main__':
    main()
