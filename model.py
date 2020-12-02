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





class Model(tf.keras.Model):
    """Model class representing your MPNN."""

    def __init__(self):
        """
        Instantiate a lifting layer, an optimizer, some number of MPLayers
        (we recommend 3), and a readout layer.
        """
        super(Model, self).__init__()

        # TODO: Initialize hyperparameters
        self.raw_features = 119
        self.num_classes = 2
        self.learning_rate = 1e-4
        self.hidden_size = 300
        self.batch_size = 10

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.liftLayer = tf.keras.layers.Dense(self.hidden_size)

        self.mp = MPLayer(self.hidden_size,self.hidden_size)
        self.mp1 = MPLayer(self.hidden_size,self.hidden_size)
        self.mp2 = MPLayer(self.hidden_size,self.hidden_size)


    def call(self, g, is_testing=False):
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

        return self.readout(g,g.ndata['node_feats'])

    def readout(self, g, node_feats):
        """
        Reduces the dimensionality of the graph to
        num_classes, and sums the node features in order to return logits.
        :param g: The batched DGL graph
        :param node_feats: The features at each node in the graph. Tensor of shape
                                   (num_atoms_in_batched_graph,
                                    size_of_node_vectors_from_prev_message_passing)
        :return: logits tensor of size (batch_size, 2)
        """
        # TODO: Set the node features to be the output of your readout layer on
        # node_feats, then use dgl.sum_nodes to return logits.
        g.ndata['node_feats'] = self.readoutLayer(node_feats)

        return dgl.sum_nodes(g,'node_feats')

    def accuracy_function(self, logits, labels):
        """
        Computes the accuracy across a batch of logits and labels.
        :param logits: a 2-D np array of size (batch_size, 2)
        :param labels: a 1-D np array of size (batch_size)
            (1 for if the molecule is active against cancer, else 0).
        :return: mean accuracy over batch.
        """
        pass


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
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.messageLayer = tf.keras.layers.Dense(self.in_feats, activation = 'relu')
        #self.reduceLayer = tf.keras.layers.Dense(self.in_feats, activation = 'relu')
        self.outputLayer = tf.keras.layers.Dense(self.out_feats, activation = 'relu')


    def call(self, g, is_testing=False):
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

    def message(self, edges, is_testing=False):
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
        #one layer for node feature weights
        #one later for edge feature weights
        #messageLayer combination of these two layers


        #or:
        #one layer -->  edge feature weight and node feature weights


        return {'msg' : self.messageLayer(edges.src['node_feats'])}

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
        edge_data.append(dict[frozenset(e)])
    edge_d_tensor = tf.convert_to_tensor(edge_data)
    #convert to one_hot tensor
    edge_oh = tf.one_hot(edge_d_tensor, 4)

    dgl_graph.ndata['node_feats'] = node_feats
    dgl_graph.edata['edge_feats'] = edge_oh



    ############### relevant to message passing #############################

    '''
    The goal here is to construct a matrix that can be multiplied by our
    "lookup" matrix in message passing. The end result is a vector of size
    (num_nodes,)

    '''


    # src and dst nodes subgraphs
    srcs = dgl_graph.subgraph(dgl_graph.edges()[0].numpy()).ndata['node_feats']
    dsts = dgl_graph.subgraph(dgl_graph.edges()[1].numpy()).ndata['node_feats']
    # based on this result, I believe that srcs and dsts are ordered how I want them
    print(np.count_nonzero(srcs.numpy() == dsts.numpy()), np.shape(srcs.numpy()), 28*119)
    # should correspond to the edges in the same ordering as our source nodes
    e_idxs = tf.argmax(dgl_graph.edata['edge_feats'], axis=1)
    s_idxs = tf.argmax(srcs, axis=1)
    d_idxs = tf.argmax(dsts, axis=1)

    # size: (num nodes, 114, 114, 4)
    multi_hots = np.zeros((28, 114, 114, 4))

    c = 0
    # find a way to vectorize this
    for s,d,b in zip(s_idxs, d_idxs, e_idxs):
        multi_hots[c, s, d, b] = 1
        c += 1

    # this verifies we have it correct, we end up with 28 total nonzero indices,
    # which is equal to the number of nodes
    print(np.count_nonzero(multi_hots), np.nonzero(multi_hots))

    var = tf.Variable(tf.random.normal([114, 114, 3], stddev=0.1))

    res = tf.matmul(multi_hots, var)

    print(np.count_nonzero(res))

    return dgl_graph


def train(model, train_data):
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

    pass




def test(model, test_data):
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
    # TODO: Fill this in!
    graphs = []
    labels = []
    for m in test_data:
        graphs.append(build_graph(m))
        labels.append(m.label)

    labels = tf.convert_to_tensor(labels)
    g_batched = dgl.batch(graphs)
    logits = model.call(g_batched, True)
    return model.accuracy_function(logits, labels)

def main():
    # TODO: Return the training and testing data from get_data
    # TODO: Instantiate model
    # TODO: Train and test for up to 15 epochs.
    train_molecules, train_labels, test_molecules, vocab = get_data('./data/test.csv', './data/train.csv', \
     './data/vocab.txt')

    print(build_graph(train_molecules[0]))



if __name__ == '__main__':
    main()
