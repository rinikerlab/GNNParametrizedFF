import os

import graph_nets as gn
import numpy as np
import sonnet as snt
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from Dispersion import BJ_dispersion
from Electrostatics import G_matrices, B_matrices, es_damped_multipole
from Exchange import aniso_exchange_potential
from Induction import induction_potential, CT
from Utilities import cdist_tf_batch, outer_product, D_Q, S, A


def load_model(model, save_path=''):
    checkpoints = []
    checkpoints.extend([tf.train.Checkpoint(module=model.embedding),
                        tf.train.Checkpoint(module=model.parameter_prediction),
                        tf.train.Checkpoint(module=model.parameter_prediction_atomic)])
    names = ['_embedding', 'parameter_prediction', 'parameter_prediction_atomic']
    for idl, layer in enumerate(model.gns):
        checkpoints.append(tf.train.Checkpoint(module=layer))
        names.append('_layer_gn_'+ str(idl))
    for idc, checkpoint in enumerate(checkpoints):
        checkpoint.restore(save_path + names[idc] + '-1')
    return model

def mila(x, beta=-1):
    return x * tf.math.tanh(tf.math.softplus(beta + x))

def shifted_softplus(x, shift=1e-3):
    return tf.nn.softplus(x) + shift

ff_module = lambda node_size, num_layers, l1=0, l2=0: \
                    tf.keras.Sequential([
                                            tf.keras.layers.Dense(
                                                units=node_size, #64
                                                activation=mila,
                                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                                kernel_initializer=tf.keras.initializers.he_normal(),
                                            ) for _ in range(num_layers)
                                        ])

ff_module_terminal = lambda node_size, num_layers, output_size, l1=0, l2=0, final_activation=shifted_softplus: \
                    tf.keras.Sequential([
                                            tf.keras.layers.Dense(
                                                units=node_size, #64
                                                activation=mila,
                                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                                kernel_initializer=tf.keras.initializers.he_normal(),
                                            ) for _ in range(num_layers)
                                        ] + \
                                        [
                                            tf.keras.layers.Dense(
                                                units=output_size, 
                                                activation=final_activation,
                                                bias_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                                kernel_initializer=tf.keras.initializers.he_normal(),
                                        )])


class ANA(tf.keras.Model):
    def __init__(self, cutoff=50.0, l1=0.0, l2=0.0, activation=mila, num_steps=3, node_size=64, edge_size=64, num_parameters=5, num_parameters_atomic=5,
                n_iter=8, omega=0.75):
        super(ANA, self).__init__()   
        self.num_steps = num_steps
        self.regularizer = snt.regularizers.L1(l1)
        self.initializer = tf.keras.initializers.he_normal()
        self.cutoff = cutoff
        self.node_size = node_size
        self.edge_size = edge_size
        self.num_parameters = num_parameters  
        self.num_parameters_atomic = num_parameters_atomic
        self.n_iter = n_iter
        self.omega = omega
        
        self.embedding = gn.modules.GraphIndependent(
                            edge_model_fn=lambda: ff_module(self.edge_size, 1, l1=1e-3, l2=l2),
                            node_model_fn=lambda: ff_module(self.node_size, 1, l1=1e-3, l2=l2),     
                        )   

        self.gns = [gn.modules.InteractionNetwork( 
                        edge_model_fn=lambda: ff_module(self.edge_size, 1, l1=l1, l2=l2),
                        node_model_fn=lambda: ff_module(self.node_size, 1, l1=l1, l2=l2))   
                         for _ in range(self.num_steps)]
        
        self.parameter_prediction = ff_module_terminal(self.node_size, 2, self.num_parameters, l1=1e-3, l2=l2)   
        self.parameter_prediction_atomic = ff_module_terminal(self.node_size, 2, self.num_parameters_atomic, l1=1e-3, l2=l2) 
            
    def _update_graphs(self, graphs):
        graphs = self.embedding(graphs)
        for layer in self.gns:
            graphs = layer(graphs)
        return graphs
    
    def _predict_parameters(self, graph_1, graph_2, indices_1, indices_2):
        graph_1, graph_2 = self._update_graphs(graph_1), self._update_graphs(graph_2)
        features_monomer_1, features_monomer_2 = graph_1.nodes, graph_2.nodes
        features_1, features_2 = tf.gather(features_monomer_1, indices_1), tf.gather(features_monomer_2, indices_2)
        prediction_1 = self.parameter_prediction(tf.concat((features_1, features_2), axis=-1))
        prediction_2 = self.parameter_prediction(tf.concat((features_2, features_1), axis=-1))
        parameters = (prediction_1 + prediction_2)
        atomic_parameters_1 = self.parameter_prediction_atomic(features_monomer_1) 
        atomic_parameters_2 = self.parameter_prediction_atomic(features_monomer_2)
        return graph_1, graph_2, parameters, atomic_parameters_1, atomic_parameters_2

    def call(self, graph_1, graph_2, coords_1, coords_2, multipoles, distance_matrices):
        h_indices_1, h_indices_2 = tf.argmax(graph_1.nodes, axis=-1) == 0, tf.argmax(graph_2.nodes, axis=-1) == 0
        indices_1, indices_2, R1, R2, Rx1, Rx2 = self._prepare_distances(distance_matrices, coords_1, coords_2)   
        graph_1, graph_2, parameters, atomic_parameters_1, atomic_parameters_2 = self._predict_parameters(graph_1, graph_2, indices_1, indices_2)
        B0, B1, B2, B3, B4 = B_matrices(R1, R2)
        alpha_damping_1, pol_1, K1, dq_ex_1, alpha_ex_1 = tf.split(atomic_parameters_1, 5, axis=-1)
        alpha_damping_2, pol_2, K2, dq_ex_2, alpha_ex_2 = tf.split(atomic_parameters_2, 5, axis=-1)
        dq_ex_1 -= multipoles[0] 
        dq_ex_2 -= multipoles[1]
        q_ex_1 = tf.where(h_indices_1[tf.newaxis, :, tf.newaxis], 1.0 - multipoles[0], 2.0 + dq_ex_1)
        q_ex_2 = tf.where(h_indices_2[tf.newaxis, :, tf.newaxis], 1.0 - multipoles[1], 2.0 + dq_ex_2)        
        in_term, mu_ind_1, mu_ind_2 = induction_potential(coords_1, coords_2, pol_1, pol_2, distance_matrices, multipoles, smear=-0.39)
        in_term += self._charge_transfer_potential(R1, parameters[..., -2:])
        disp_term = self._dispersion_potential(R2, parameters[..., :3])
        es_term = es_damped_multipole(R1, R2, Rx1, Rx2, B0, B1, B2, B3, B4,
                                      multipoles, alpha_damping_1, alpha_damping_2, indices_1, indices_2)
        ex_term = aniso_exchange_potential(R1, R2, Rx1, B0, B1, B2, B3, B4, 
                                           q_ex_1, q_ex_2, alpha_ex_1, alpha_ex_2, K1, K2, 
                                           multipoles, mu_ind_1, mu_ind_2, indices_1, indices_2)
        return ex_term + disp_term + es_term + in_term, es_term, in_term, ex_term, disp_term
    
    def _prepare_distances(self, distance_matrices, coords_1, coords_2):
        n_nodes_1, n_nodes_2 = distance_matrices.shape[1:3]
        interaction_indices = np.indices((n_nodes_1, n_nodes_2)).reshape((2, -1)).T
        indices_1, indices_2 = tf.unstack(interaction_indices, axis=-1)
        R1 = tf.reshape(distance_matrices, [distance_matrices.shape[0], -1, 1])
        R2 = tf.square(R1)
        Rx1 = tf.gather(coords_2, indices_2, batch_dims=0, axis=1) - tf.gather(coords_1, indices_1, batch_dims=0, axis=1)
        Rx2 = outer_product(Rx1)
        return indices_1, indices_2, R1, R2, Rx1, Rx2
        
    def _dispersion_potential(self, R2, parameters):
        parameters = tf.tile(tf.expand_dims(parameters, axis=0), [R2.shape[0], 1, 1])
        C6, C8, C10 = tf.split(parameters, 3, axis=-1)
        disp_term = tf.where(R2 < tf.square(self.cutoff), BJ_dispersion(R2, C6, C8, C10), 0)
        return tf.reduce_sum(disp_term, axis=[-1, -2])
    
    def _charge_transfer_potential(self, R1, parameters):        
        A, b = tf.split(parameters, 2, axis=-1)
        return -tf.reduce_sum(CT(R1, A, b), axis=[-1, -2])
