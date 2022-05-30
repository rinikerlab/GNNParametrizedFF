import tensorflow as tf

from Electrostatics import G_matrices
from Utilities import outer_product

# Anisotropic repulsion from 'Classical Pauli repulsion: An anisotropic, atomic multipole model' by Rackers et al.
# https://aip.scitation.org/doi/10.1063/1.5081060
# Damping coefficients are based on the reference implementation found in 
# https://github.com/JoshRackers/tinker/blob/amoeba2/source/damping.f
def damping_ex_diff(apauli, apaulk, R1, R2, R3, R4, R5):  
    apauli2, apaulk2 = 0.5 * apauli, 0.5 * apaulk
    apauli2_SQ, apaulk2_SQ = apauli2 ** 2, apaulk2 ** 2
    apauli2_CU, apaulk2_CU = apauli2_SQ * apauli2, apaulk2_SQ * apaulk2
    apauli2_HY, apaulk2_HY = apauli2_SQ * apauli2_SQ, apaulk2_SQ * apaulk2_SQ
    X = apauli2_SQ - apaulk2_SQ
    R6 = R5 * R1
    apaulk2_X, apaulk2_SQ_X = apaulk2 / X, apaulk2_SQ / X
    dampi, dampk = apauli2 * R1, apaulk2 * R1
    expdampi, expdampk = tf.math.exp(-dampi), tf.math.exp(-dampk)
    PRE = 64.0 * (apauli**3) * (apaulk**3) / (X ** 4)
    a2_product = apaulk2 * apauli2        
    S = (dampi - 0.4E1 / X * a2_product) * expdampk + (dampk + 0.4E1 / X * a2_product) * expdampi
    DS = (apauli2 * apaulk2 * R2 - 0.4E1 * apauli2 * apaulk2_SQ_X * R1 - 0.4E1 / X * a2_product) * expdampk +\
        (apauli2 * apaulk2 * R2 + 0.4E1 / X * a2_product + 0.4E1 * apauli2_SQ * apaulk2_X * R1) * expdampi
    DDS = (apauli2 * apaulk2 * R2 / 0.3E1 - 0.4E1 * apauli2 * apaulk2_SQ_X * R1 + apauli2 * apaulk2_SQ * R3 / 0.3E1 - \
           0.4E1 / 0.3E1 * apauli2 * apaulk2_CU / X * R2 - 0.4E1 / X * a2_product) * expdampk +\
        (apauli2 * apaulk2 * R2 / 0.3E1 + 0.4E1 / X * a2_product + apauli2_SQ * apaulk2 * R3 / 0.3E1 +\
         0.4E1 * apauli2_SQ * apaulk2_X * R1 + 0.4E1 / 0.3E1 * apauli2_CU * apaulk2_X * R2) * expdampi
    DDDS = (-0.4E1 / X * a2_product - 0.4E1 * apauli2 * apaulk2_SQ_X * R1\
            - 0.8E1 / 0.5E1 * apauli2 * apaulk2_CU / X * R2\
            + apauli2 * apaulk2 * R2 / 0.5E1 + apauli2 * apaulk2_SQ * R3 / 0.5E1\
            + apauli2 * apaulk2_CU * R4 / 0.15E2\
            - 0.4E1 / 0.15E2 * apauli2 * apaulk2_HY / X * R3) * expdampk\
            + (apauli2_CU * apaulk2 * R4 / 0.15E2\
            + 0.4E1 / 0.15E2 * apauli2_HY * apaulk2_X * R3\
            + 0.4E1 / X * a2_product + 0.4E1 * apauli2_SQ * apaulk2_X * R1\
            + 0.8E1 / 0.5E1 * apauli2_CU * apaulk2_X * R2 + apauli2 * apaulk2 * R2 / 0.5E1\
            + apauli2_SQ * apaulk2 * R3 / 0.5E1) * expdampi    
    DDDDS = (-0.12E2 / 0.7E1 * apauli2 * apaulk2_CU / X * R2 \
             - 0.8E1 / 0.21E2 * apauli2 * apaulk2_HY / X * R3\
             + apauli2 * apaulk2 * R2 / 0.7E1 + apauli2 * apaulk2_SQ * R3 / 0.7E1\
             - 0.4E1 / X * a2_product + 0.2E1 / 0.35E2 * apauli2 * apaulk2_CU * R4\
             - 0.4E1 / 0.105E3 * apauli2 * apaulk2 ** 5 / X * R4\
             + apauli2 * apaulk2_HY * R5 / 0.105E3\
             - 0.4E1 * apauli2 * apaulk2_SQ_X * R1) * expdampk\
             + (apauli2_SQ * apaulk2 * R3 / 0.7E1 + 0.4E1 / X * a2_product\
             + 0.2E1 / 0.35E2 * apauli2_CU * apaulk2 * R4\
             + 0.4E1 / 0.105E3 * apauli2 ** 5 * apaulk2_X * R4\
             + apauli2_HY * apaulk2 * R5 / 0.105E3 + 0.4E1 * apauli2_SQ * apaulk2_X * R1\
             + 0.12E2 / 0.7E1 * apauli2_CU * apaulk2_X * R2\
             + 0.8E1 / 0.21E2 * apauli2_HY * apaulk2_X * R3\
             + apauli2 * apaulk2 * R2 / 0.7E1) * expdampi
    return PRE, S, DS, DDS, DDDS, DDDDS

def damping_ex_same(apauli, apaulk, R1, R2, R3, R4, R5):
    apauli2, apaulk2 = 0.5 * apauli, 0.5 * apaulk
    apauli = tf.math.minimum(apauli, apaulk)
    apauli2 = tf.math.minimum(apauli2, apaulk2)
    dampi = apauli2*R1
    expdampi = tf.math.exp(-dampi)
    PRE = (apauli**6)/(apauli2**6) 
    R6 = R5 * R1
    S = (R1 + R2 * apauli2 + R3 * apauli2 ** 2 / 0.3E1) * expdampi
    DS = (R3 * apauli2 ** 2 / 0.3E1 + R4 * apauli2 ** 3 / 0.3E1) * expdampi
    DDS = apauli2 ** 4 * expdampi * R5 / 0.9E1
    DDDS = apauli2 ** 5 * expdampi * R6 / 0.45E2
    DDDDS = (apauli2 ** 5 * R6 / 0.315E3 + apauli2 ** 6 * R6 * R1 / 0.315E3) * expdampi
    return PRE, S, DS, DDS, DDDS, DDDDS

def aniso_exchange_potential(R1, R2, Rx1, B0, B1, B2, B3, B4, q_ex_1, q_ex_2, 
                             alpha_ex_1, alpha_ex_2, K1, K2, multipoles, mu_ind_1, mu_ind_2, indices_1, indices_2):
    _, _, dipoles_1, dipoles_2, quadrupoles_1, quadrupoles_2 = multipoles[:6]
    dipoles_1 += mu_ind_1
    dipoles_2 += mu_ind_2
    q_ex_1, q_ex_2 = tf.gather(q_ex_1, indices_1, axis=1), tf.gather(q_ex_2, indices_2, axis=1)
    dipos_1, dipos_2 = tf.gather(dipoles_1, indices_1, axis=1), tf.gather(dipoles_2, indices_2, axis=1)
    quads_1, quads_2 = tf.gather(quadrupoles_1, indices_1, axis=1), tf.gather(quadrupoles_2, indices_2, axis=1)
    K_products = (tf.gather(K1, indices_1) * tf.gather(K2, indices_2))[tf.newaxis]
    alpha_ex_1, alpha_ex_2 = tf.gather(alpha_ex_1, indices_1), tf.gather(alpha_ex_2, indices_2)
    alpha_ex_1, alpha_ex_2 = tf.tile(alpha_ex_1[tf.newaxis], [R1.shape[0], 1, 1]), tf.tile(alpha_ex_2[tf.newaxis], [R1.shape[0], 1, 1])
    alpha_neq_condition = tf.math.abs(alpha_ex_1 - alpha_ex_2) > 1e-2
    R3, R4 = R2 * R1, R2 * R2
    R5 = R4 * R1
    Rx2 = outer_product(Rx1)
    PRE, S, DS, DDS, DDDS, DDDDS = tf.where(alpha_neq_condition, damping_ex_diff(alpha_ex_1, alpha_ex_2, R1, R2, R3, R4, R5), 
                                                                 damping_ex_same(alpha_ex_1, alpha_ex_2, R1, R2, R3, R4, R5))
    G0, G1, G2, G3, G4 = G_matrices(q_ex_1, q_ex_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2)
    S = S * B0
    DS = DS * B1
    DDS = DDS * B2
    DDDS = DDDS * B3
    DDDDS = DDDDS * B4
    DS_DDS = DS * DDS
    DS_DDDS = DS * DDDS
    DDS_SQ = DDS * DDS
    L0 = PRE * S ** 2
    L1 = 2.0 * PRE * S * DS
    L2 = 2.0 * PRE * (DS * DS + S * DDS)
    L3 = 2.0 * PRE * (DS_DDS + DS_DDS + DS_DDS + S * DDDS)
    L4 = 2.0 * PRE * (DDDS * DS + DDS_SQ + DDS_SQ + DS_DDDS + DDS_SQ + DS_DDDS + DS_DDDS + S * DDDDS)
    S2 = G0 * L0 + G1 * L1 + G2 * L2 + G3 * L3 + G4 * L4
    return tf.reduce_sum(K_products * S2 * B0, axis=[-1, -2]) 