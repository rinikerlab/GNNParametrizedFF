import graph_nets as gn
import numpy as np
import tensorflow as tf

from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import AddHs
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree


class GraphBuilder:  
    def __init__(self, dtype_np=np.float32, dtype_tf=tf.float32):
        self._dtype_np = dtype_np
        self._dtype_tf = dtype_tf
        self.ONEHOTS = {
                                        'H': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=self._dtype_np),
                                        'C': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=self._dtype_np),
                                        'N': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=self._dtype_np),
                                        'O': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=self._dtype_np),                        
                                        'F': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=self._dtype_np),
                                        'P': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=self._dtype_np),
                                        'S': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=self._dtype_np),
                                        'Cl': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=self._dtype_np),  
                                        'Br': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=self._dtype_np),    
                                        'I': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=self._dtype_np),
                                        'CL': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),   
                                        'BR': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
                                    }
        
        self.REF_BOND_LENGTHS = {
            'C': 2.0, # C-S
            'N': 1.8, # N-S
            'O': 1.8, # O-S
            'S': 2.25, # Pure Sulfur Compounds may be bigger

        }

        self.REF_MAX_NEIGHBOURS = {
            'C': 4,
            'N': 4,
            'O': 2,
            'S': 4,
        }
        
    def _build_graph(self, elements, senders, receivers, edge_features=None):
        senders, receivers = senders.astype(np.int32), receivers.astype(np.int32)
        node_features = np.vstack([self.ONEHOTS[e]  for e in elements])
        if edge_features is None:
            edge_features = np.concatenate((node_features[senders], node_features[receivers]), axis=-1)
        n_node, n_edge = np.int32(len(node_features)), np.int32(len(edge_features))
        return gn.graphs.GraphsTuple(node_features, edge_features, globals=None, receivers=receivers, senders=senders, n_node=n_node, n_edge=n_edge)
    
    def from_smile(self, smile):
        return self.from_mol(AddHs(MolFromSmiles(smile)))

    def from_mol(self, mol):
        elements = [atom.GetSymbol()  for atom in mol.GetAtoms()]
        start, end = np.array([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]).T
        senders, receivers = np.hstack((start, end)), np.hstack((end, start))
        return self._build_graph(elements, senders, receivers)
    
    def topological_from_coords(self, coordinates, elements):
        bonds = self._bonds_from_coords(coordinates, elements)
        return self._build_graph(elements, bonds[..., 0], bonds[..., 1])
    
    def _bonds_from_coords(self, coordinates, elements):
        single_valence_condition = (elements == 'H') | (elements == 'F') | (elements == 'Cl') | (elements == 'Br') | (elements == 'I')
        single_valence_indices = np.where(single_valence_condition)[0]
        multiple_valence_indices = np.where(~single_valence_condition)[0]        
        kd_tree = KDTree(coordinates)        
        neighbours_single_valence = kd_tree.query(coordinates[single_valence_indices], k=[2])[1][:, 0]     
        multi_valence_eles = elements[multiple_valence_indices]
        bond_lengths = [self.REF_BOND_LENGTHS[e] for e in multi_valence_eles]
        #max_neighs = [self.REF_MAX_NEIGHBOURS[e] for e in multi_valence_eles]        
        neighbours_multi_valence = kd_tree.query_ball_point(coordinates[multiple_valence_indices], bond_lengths) 
        bonds = []
        for current_atom_index, current_element, neighbour_indices in zip(multiple_valence_indices, multi_valence_eles, neighbours_multi_valence):
            #neighbour_indices = neighbour_indices[1:REF_MAX_NEIGHBOURS[current_element]]
            for neighbour_index in neighbour_indices:
                if neighbour_index not in single_valence_indices and neighbour_index != current_atom_index:
                    bonds.append((current_atom_index, neighbour_index))                    
        single_valence_pairs = np.stack((neighbours_single_valence, single_valence_indices), axis=-1)
        single_valence_pairs = np.vstack((single_valence_pairs, np.flip(single_valence_pairs, axis=-1)))
        #print(np.array(bonds).shape, single_valence_pairs.shape)
        bonds = np.array(bonds)
        if bonds.size > 0 and single_valence_pairs.size > 0:        
            bonds = np.vstack((bonds, single_valence_pairs))
        elif bonds.size == 0 and single_valence_pairs.size > 0:
            bonds = np.array(single_valence_pairs)
        #_, unique_indices = np.unique(np.stack((graph_1.receivers, graph_1.senders)).T, axis=0, return_index=True)
        return np.unique(bonds, axis=0)  
    
    def geometric_from_coords(self, coords, elements, cutoff=4.0, num_kernels=32):
        dmat = squareform(pdist(coords))
        indices = tf.where(tf.math.logical_and(dmat > 0, dmat < cutoff))
        edge_weights = weight_function(tf.expand_dims(tf.gather_nd(dmat, indices), axis=-1), num_kernels, 0.5, cutoff)[0]
        return self._build_graph(elements, indices[..., 0], indices[..., 1], edge_features=edge_weights)
