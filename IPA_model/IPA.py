import os

import graph_nets as gn
import numpy as np
import sonnet as snt
import tensorflow as tf


from GraphBuilder import GraphBuilder


def mila(x, beta=-1):
    return x * tf.math.tanh(tf.math.softplus(beta + x))

class IPA(tf.keras.Model):
    def __init__(self, cutoff=10.0, l1=0.0, l2=0.0, activation=mila, num_steps=3, node_size=64, edge_size=64, num_parameters=2):
        super(IPA, self).__init__()   
        self.num_steps = num_steps
        self.regularizer = snt.regularizers.L1(l1)
        self.initializer = tf.keras.initializers.he_normal()
        self.cutoff = cutoff
        self.node_size = node_size
        self.edge_size = edge_size
        self.num_parameters = num_parameters        
        self.builder = GraphBuilder()
        
        self.embedding = gn.modules.GraphIndependent(
                                edge_model_fn=lambda: snt.nets.MLP(
                                                            output_sizes=[self.edge_size],
                                                            w_init=self.initializer,
                                                            with_bias=True,
                                                            activation=activation,
                                                            activate_final=True,
                                                        ),
                                node_model_fn=lambda: snt.nets.MLP(
                                                            output_sizes=[self.node_size],
                                                            w_init=self.initializer,
                                                            with_bias=True,
                                                            activation=activation,
                                                            activate_final=True,        
                                                        ),
                                                    )   

        self.gns = [gn.modules.InteractionNetwork(
                        edge_model_fn=lambda: snt.nets.MLP(
                                                    output_sizes=[self.edge_size, self.edge_size],
                                                    w_init=self.initializer,
                                                    with_bias=True,
                                                    activation=activation,
                                                    activate_final=True,
                                                ),
                        node_model_fn=lambda: snt.nets.MLP(
                                                    output_sizes=[self.node_size, self.node_size],
                                                    w_init=self.initializer,
                                                    with_bias=True,
                                                    activation=activation,
                                                    activate_final=True,        
                                                ),
                                            ) for _ in range(self.num_steps)]
        
        self.parameter_prediction = tf.keras.Sequential([
                                                            tf.keras.layers.Dense(
                                                                units=self.node_size, 
                                                                activation=activation,
                                                                kernel_initializer=self.initializer,
                                                            ),
                                                            tf.keras.layers.Dense(
                                                                units=self.node_size, 
                                                                activation=activation,
                                                                kernel_initializer=self.initializer,
                                                            ),
                                                            tf.keras.layers.Dense(
                                                                units=self.num_parameters, 
                                                                activation=tf.nn.softplus, 
                                                                kernel_initializer=self.initializer,
                                                            ),
                                                        ])
            
    def _update_graphs(self, graphs):
        graphs = self.embedding(graphs)
        for layer in self.gns:
            graphs = layer(graphs)
        return graphs

    def predict_parameters(self, coordinates, elements):
        graph = self.builder.topological_from_coords(coordinates, elements)
        interaction_indices = np.indices((graph.n_node, graph.n_node)).reshape((2, -1)).T
        indices_1, indices_2 = tf.unstack(interaction_indices, axis=-1)
        _, _, parameters = self._predict_parameters(graph, graph, indices_1, indices_2)
        return parameters, interaction_indices    
    
    def predict_parameters_from_graph(self, graph):
        interaction_indices = np.indices((graph.n_node, graph.n_node)).reshape((2, -1)).T
        indices_1, indices_2 = tf.squeeze(tf.split(interaction_indices, 2, axis=-1))
        _, _, parameters = self._predict_parameters(graph, graph, indices_1, indices_2)
        return parameters, interaction_indices    
    
    def _predict_parameters(self, graph_1, graph_2, indices_1, indices_2):
        graph_1, graph_2 = self._update_graphs(graph_1), self._update_graphs(graph_2)
        features_monomer_1, features_monomer_2 = graph_1.nodes, graph_2.nodes
        features_1, features_2 = tf.gather(features_monomer_1, indices_1), tf.gather(features_monomer_2, indices_2)
        prediction_1 = self.parameter_prediction(tf.concat((features_1, features_2), axis=-1))
        prediction_2 = self.parameter_prediction(tf.concat((features_2, features_1), axis=-1))
        parameters = (prediction_1 + prediction_2)
        return graph_1, graph_2, parameters

    def call(self, graph_1, graph_2, distance_matrices):
        indices_1, indices_2, R1 = self._prepare_distances(distance_matrices)   
        graph_1, graph_2, parameters = self._predict_parameters(graph_1, graph_2, indices_1, indices_2)
        ex_term, disp_term = self._dispersion_exchange_potential(R1, parameters)
        return ex_term + disp_term, ex_term, disp_term
    
    def _prepare_distances(self, distance_matrices):
        n_batches, n_nodes_1, n_nodes_2 = distance_matrices.shape[:3]
        interaction_indices = np.indices((n_nodes_1, n_nodes_2)).reshape((2, -1)).T
        indices_1, indices_2 = tf.unstack(interaction_indices, axis=-1)
        R1 = tf.reshape(distance_matrices, [n_batches, -1, 1])
        return indices_1, indices_2, R1
    
    def _dispersion_exchange_potential(self, R1, parameters):
        parameters = tf.tile(tf.expand_dims(parameters, axis=0), [R1.shape[0], 1, 1])
        C6, C9 = tf.split(parameters, 2, axis=-1)
        ex_term, disp_term = tf.where(R1 < self.cutoff, C6C9(R1, C6, C9), 0)
        return tf.reduce_sum(ex_term, axis=[-1, -2]), tf.reduce_sum(disp_term, axis=[-1, -2])    
    
def C6C9(R1, C6, C9):
    R3 = tf.math.pow(R1, 3)
    R6 = R3 * R3
    R9 = R6 * R3
    return C9 / R9, -C6 / R6
    
def load_model(model=None, save_path='', num_steps=3, node_size=64, edge_size=64, num_params=2, l1=0):
    if model is None:
        model = IPA(num_steps=num_steps, edge_size=edge_size, node_size=node_size, num_parameters=num_params, l1=l1)
    checkpoints = []
    checkpoints.extend([tf.train.Checkpoint(module=model.embedding),
                        tf.train.Checkpoint(module=model.parameter_prediction),])
    names = ['_embedding', '_parameter_prediction',]
    for idl, layer in enumerate(model.gns):
        checkpoints.append(tf.train.Checkpoint(module=layer))
        names.append('_layer_gn_'+ str(idl))
    for idc, checkpoint in enumerate(checkpoints):
        checkpoint.restore(save_path + names[idc] + '-1')
    return model

def cdist_tf_batch(A, B):
    na = tf.reduce_sum(tf.square(A), axis=-1, keepdims=True)
    nb = tf.transpose(tf.reduce_sum(tf.square(B), axis=-1, keepdims=True), [0, 2, 1])
    return tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))