from preprocess import get_data
from tensorflow.keras.layers import Dense, Layer
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
        # TODO: Fill this in!
        pass

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
        pass

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
        pass


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
        pass

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
        pass

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
        pass


def build_graph(smiles):
    """
    Constructs a NetworkX graph out of a SMILES representation of a molecule from the train/test data.
    :param smiles: a string object of SMILES format
    :return: nx.Graph
        A graph describing a molecule. Nodes will have an 'element', 'aromatic'
        and a 'charge', and if `explicit_hydrogen` is False a 'hcount'.
        Depending on the input, they will also have 'isotope' and 'class'
        information.
        Edges will have an 'order'.
    """
    # TODO: Initialize a DGL Graph
    print(smiles)
    g = pysmiles.read_smiles(smiles)
    #edge_att = g.get_edge_data()
    node_feats = g.nodes(data='element')
    na = np.array(node_feats)
    print(na)
    t1 =tf.convert_to_tensor(na[:,1])
    print("ft",t1)


    
    edata = g.edges(data='order')
    na = np.array(edata)
    #print(na)
    #t2 = tf.convert_to_tensor(na[:,:,0])
    #print("t2", t2)
    #nd = tf.convert_to_tensor()
    #ed = tf.convert_to_tensor(g.edges(data='order'))
    #sprint("node",g.nodes(data='element'))
    #print("edge",g.edges(data='order'))
    #print("edge",e.edata)
    dgl_graph = dgl.from_networkx(g)
    dgl_graph.ndata['node_features'] = nd
    dgl_graph.edata['edge_features'] = ed
    print("ndata",dgl.graph.ndata)
    print("edata",dgl.graph.edata)
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
    train_molecules, train_labels, test_molecules, vocab = get_data('.././data/test.csv', '.././data/train.csv', \
     '.././data/vocab.txt')

    print(build_graph(train_molecules[0]))



if __name__ == '__main__':
    main()
