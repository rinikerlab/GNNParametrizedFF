import tensorflow as tf
import numpy as np
from Utilities import cdist_tf_batch, A

KC = 1389.35457644382 
permutation_mask = np.array([-1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
mask = []
for x in range(3):
    for y in range(3):
        for z in range(3):
            base = np.zeros((3))
            if y == z:
                base[x] += 1
            if x == z:
                base[y] += 1
            if x == y:
                base[z] += 1
            mask.append(base)
mask = np.array(mask)

# Short-range charge-transfer/induction potential based on the Amoeba+ model:
# https://pubs.acs.org/doi/10.1021/acs.jctc.9b00261
def CT(R, A, b):
    return A * tf.exp(-b * R)


# Thole model used to generate induced dipoles.
# https://www.sciencedirect.com/science/article/pii/0301010481851762
def get_T_field(coords_1, coords_2, dmats):
    dmats_sq = tf.square(dmats)
    dmats_3 = dmats_sq * dmats
    dmats_5 = dmats_sq * dmats_3
    dmats_7 = dmats_sq * dmats_5
    RxR = coords_2[:, tf.newaxis] - coords_1[:, :, tf.newaxis]
    
    T_q_mu = tf.math.divide_no_nan(-RxR, dmats_3)
    T_mu_mu = tf.math.divide_no_nan(3 * (tf.expand_dims(RxR, axis=-1) * tf.expand_dims(RxR, axis=-2)), dmats_5[..., tf.newaxis])
    T_mu_mu_diag = tf.math.divide_no_nan(tf.reshape(tf.eye(3), [1, 1, 1, 3, 3]), dmats_3[..., tf.newaxis])
    T_mu_mu -= T_mu_mu_diag
    T_t_mu = -tf.math.divide_no_nan(
                (15 * RxR[:, :, :, :, tf.newaxis, tf.newaxis] * \
                      RxR[:, :, :, tf.newaxis, :, tf.newaxis] * \
                      RxR[:, :, :, tf.newaxis, tf.newaxis, :]), 
                      dmats_7[..., tf.newaxis, tf.newaxis])
    T_t_mu = tf.reshape(T_t_mu, (*T_t_mu.shape[:4], 9))
    T_t_mu_diag = tf.math.divide_no_nan(3 * tf.einsum('ijkl, ml -> ijkm', RxR, mask), dmats_5)
    T_t_mu += tf.reshape(T_t_mu_diag, (*dmats.shape[:3], 3, 9)) 
    T1 = tf.concat((T_q_mu[..., tf.newaxis], T_mu_mu, T_t_mu), axis=-1)
    T2 = tf.transpose(T1 * permutation_mask, [0, 2, 1, 3, 4])
    return T1, T2

def damping_thole(dmats, alpha_1, alpha_2, smear=-0.39):
    U = dmats / tf.math.pow(alpha_1 * tf.linalg.matrix_transpose(alpha_2), 1/6)[tf.newaxis, :, :, tf.newaxis]
    exponent = smear * tf.math.pow(U, 3)
    coeff = tf.exp(exponent)
    L3 = 1.0 - coeff
    L5 = 1.0 - (1.0 - exponent) * coeff
    L7 = 1.0 - (1.0 - exponent + 0.6 * exponent * exponent) * coeff
    return L3, L5, L7

def get_T_thole(R1, R2, R5, RxR, alphas, smear):
    batch_size, n_atoms = R1.shape[:2]
    L3, L5, L7 = damping_thole(R1[..., tf.newaxis], alphas, alphas, smear=smear)
    R2_term = L3[..., tf.newaxis] * R2[..., tf.newaxis, tf.newaxis] * tf.eye(3, dtype=tf.float32)[tf.newaxis, tf.newaxis]
    RxR_term = -3 * L5[..., tf.newaxis] * RxR
    T_thole = tf.math.divide_no_nan(R2_term + RxR_term, R5[..., tf.newaxis, tf.newaxis])
    T_thole = tf.transpose(T_thole, [0, 1, 3, 2, 4])
    T_thole = tf.reshape(T_thole, (batch_size, 3 * n_atoms, 3 * n_atoms))
    T_thole += tf.linalg.diag(tf.repeat(1 / alphas[:, 0], 3, axis=-1))[tf.newaxis]
    return T_thole

def prepare_Rs(coords):
    Rs = A(coords, [2]) - A(coords, [1])
    R1 = cdist_tf_batch(coords, coords)
    R2 = tf.square(R1)
    R5 = R2 * R2 * R1
    RxR = A(Rs, [-1]) * A(Rs, [-2])
    return R1, R2, R5, RxR

def induction_potential(coords_1, coords_2, pol_1, pol_2, dmats, multipoles, smear=-0.39):
    monopoles_1, monopoles_2, dipoles_1, dipoles_2, quadrupoles_1, quadrupoles_2 = multipoles[:6]
    M1 = tf.concat((monopoles_1, dipoles_1, tf.reshape(quadrupoles_1, [*quadrupoles_1.shape[:2], 9])), axis=-1)
    M2 = tf.concat((monopoles_2, dipoles_2, tf.reshape(quadrupoles_2, [*quadrupoles_2.shape[:2], 9])), axis=-1)
    alphas = tf.concat((pol_1, pol_2), axis=0)
    coords = tf.concat((coords_1, coords_2), axis=1)
    batch_size, n_atoms_1 = coords_1.shape[:2]
    R1, R2, R5, RxR = prepare_Rs(coords)
    B = tf.linalg.inv(get_T_thole(R1, R2, R5, RxR, alphas, smear=smear))    
    T1, T2 = get_T_field(coords_1, coords_2, dmats[..., None])
    F1, F2 = tf.einsum('bnijk, bik -> bnj', T1, M2), tf.einsum('bnijk, bik -> bnj', T2, M1)
    F = tf.reshape(tf.concat((F1, F2), axis=1), [batch_size, -1])
    mu_induced = tf.einsum('bij, bj->bi', B, F)    
    in_term = -0.5 * KC * tf.reduce_sum(mu_induced * F, axis=-1)
    mu_induced = tf.reshape(mu_induced, [batch_size, -1, 3])
    mu_ind_1, mu_ind_2 = mu_induced[:, :n_atoms_1], mu_induced[:, n_atoms_1:]
    return in_term, mu_ind_1, mu_ind_2