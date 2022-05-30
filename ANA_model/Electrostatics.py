import tensorflow as tf
from Utilities import S, A, outer_product

THIRD = 1 / 3
KC = 1389.35457644382 

# G Tensors as defined in 'A New Relatively Simple Approach to Multipole Interactions in 
# Either Spherical Harmonics or Cartesians, Suitable for Implementation into Ewald Sums '
# by Burnham et al
# https://www.mdpi.com/1422-0067/21/1/277/htm
def G_matrices(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2):
    D1_Rx1, D2_Rx1 = S(dipos_1, Rx1), S(dipos_2, Rx1)
    Q1_Rx1, Q2_Rx1 = tf.einsum('bijk, bik -> bij', quads_1, Rx1), tf.einsum('bijk, bik -> bij', quads_2, Rx1)
    Q1_Rx2, Q2_Rx2 = A(tf.einsum('bijk, bijk -> bi', quads_1, Rx2)),  A(tf.einsum('bijk, bijk -> bi', quads_2, Rx2))
    dipo_mono = D1_Rx1 * monos_2 # G1
    mono_dipo = D2_Rx1 * monos_1
    dipo_dipo = S(dipos_1, dipos_2) # G1
    dipo_R = D1_Rx1 * D2_Rx1
    G0 = monos_1 * monos_2
    G1 = dipo_mono - mono_dipo + dipo_dipo
    G2 = -dipo_R + 2 * S(Q1_Rx1, dipos_2) - 2 * S(Q2_Rx1, dipos_1)\
         + Q1_Rx2 * monos_2 + Q2_Rx2 * monos_1\
         + 2 * A(tf.einsum('bijk, bijk -> bi', quads_1, quads_2)) # bnxy, bnxy -> bn
    G3 = -4 * S(Q1_Rx1, Q2_Rx1) - Q1_Rx2 * D2_Rx1 + Q2_Rx2 * D1_Rx1
    G4 = Q1_Rx2 * Q2_Rx2
    return G0, G1, G2, G3, G4

def B_matrices(R1, R2):
    B0 = 1 / R1
    B1 = B0 / R2
    B2 = 3 * B1 / R2
    B3 = 5 * B2 / R2
    B4 = 7 * B3 / R2
    return B0, B1, B2, B3, B4

def es_multipole(R1, R2, Rx1, Rx2, B0, B1, B2, B3, B4, multipoles, indices_1, indices_2):
    monopoles_1, monopoles_2, dipoles_1, dipoles_2, quadrupoles_1, quadrupoles_2 = multipoles[:6]
    monos_1, monos_2 = tf.gather(monopoles_1, indices_1, axis=1), tf.gather(monopoles_2, indices_2, axis=1)
    dipos_1, dipos_2 = tf.gather(dipoles_1, indices_1, axis=1), tf.gather(dipoles_2, indices_2, axis=1)
    quads_1, quads_2 = tf.gather(quadrupoles_1, indices_1, axis=1), tf.gather(quadrupoles_2, indices_2, axis=1)    
    G0, G1, G2, G3, G4 = G_matrices(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2)
    N0_term = tf.reduce_sum(G0 * B0, axis=-2) 
    N1_term = tf.reduce_sum(G1 * B1, axis=-2)
    N2_term = tf.reduce_sum(G2 * B2, axis=-2)  
    N3_term = tf.reduce_sum(G3 * B3, axis=-2)  
    N4_term = tf.reduce_sum(G4 * B4, axis=-2)  
    return (N0_term + N1_term + N2_term + N3_term + N4_term)[..., 0] * KC

# Short-Range Electrostatic Model from 'Optimized Charge Penetration Model for AMOEBA' by Rackers et al: 
# https://www.rsc.org/suppdata/c6/cp/c6cp06017j/c6cp06017j1.pdf
def es_damped_multipole(R1, R2, Rx1, Rx2, B0, B1, B2, B3, B4, multipoles, alpha_1, alpha_2, indices_1, indices_2):
    monopoles_1, monopoles_2, dipoles_1, dipoles_2, quadrupoles_1, quadrupoles_2, core_charges_1, core_charges_2 = multipoles
    monopoles_1, monopoles_2 = monopoles_1 - core_charges_1, monopoles_2 - core_charges_2
    a_1, a_2 = tf.gather(alpha_1, indices_1), tf.gather(alpha_2, indices_2)
    Z_1, Z_2 = tf.gather(core_charges_1, indices_1), tf.gather(core_charges_2, indices_2)
    monos_1, monos_2 = tf.gather(monopoles_1, indices_1, axis=1), tf.gather(monopoles_2, indices_2, axis=1)
    dipos_1, dipos_2 = tf.gather(dipoles_1, indices_1, axis=1), tf.gather(dipoles_2, indices_2, axis=1)
    quads_1, quads_2 = tf.gather(quadrupoles_1, indices_1, axis=1), tf.gather(quadrupoles_2, indices_2, axis=1)    
    L0_1, L1_1, L2_1, L0_2, L1_2, L2_2, D0_pair, D1_pair, D2_pair, D3_pair, D4_pair = damping_coefficients_es(a_1, a_2, R1, R2)
    core_core = Z_1 * Z_2 * B0
    multipole_core = monos_1 * Z_2 * B0 * L0_1 + \
                     S(dipos_1, Rx1) * Z_2 * B1 * L1_1 + \
                     A(tf.einsum('bijk, bijk -> bi', quads_1, Rx2)) * Z_2 * B2 * L2_1
    core_multipole = Z_1 * monos_2 * B0 * L0_2 - \
                     S(dipos_2, Rx1) * Z_1 * B1 * L1_2 + \
                     A(tf.einsum('bijk, bijk -> bi', quads_2, Rx2)) * Z_1 * B2 * L2_2
    G0, G1, G2, G3, G4 = G_matrices(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2)
    core_core = tf.reduce_sum(core_core, axis=1)
    multipole_core = tf.reduce_sum(multipole_core, axis=1)
    core_multipole = tf.reduce_sum(core_multipole, axis=1)
    N0_term = tf.reduce_sum(G0 * B0 * D0_pair, axis=-2) 
    N1_term = tf.reduce_sum(G1 * B1 * D1_pair, axis=-2)
    N2_term = tf.reduce_sum(G2 * B2 * D2_pair, axis=-2)  
    N3_term = tf.reduce_sum(G3 * B3 * D3_pair, axis=-2)  
    N4_term = tf.reduce_sum(G4 * B4 * D4_pair, axis=-2)  
    multipole_multipole = (N0_term + N1_term + N2_term + N3_term + N4_term) 
    return (multipole_core + core_core + core_multipole + multipole_multipole)[..., 0] * KC

def damping_coefficients_es(a_1, a_2, R1, R2):
    a_1_SQ, a_2_SQ = tf.square(a_1), tf.square(a_2)
    A = a_1_SQ / (a_1_SQ - a_2_SQ)
    B = a_2_SQ / (a_2_SQ - a_1_SQ)
    A = tf.where(tf.math.abs(a_1 - a_2) < 1e-5, 1.0, A)
    B = tf.where(tf.math.abs(a_1 - a_2) < 1e-5, 1.0, B)
    A1D, A2D = a_1 * R1, a_2 * R1
    A1D_SQ, A2D_SQ = tf.math.square(A1D), tf.math.square(A2D)
    A1_EXP, A2_EXP = tf.exp(-A1D), tf.exp(-A2D)
    L0_1 = 1.0 - A1_EXP
    L1_1 = 1.0 - (1.0 + A1D) * A1_EXP
    L2_1 = 1.0 - (1.0 + A1D + A1D_SQ) * A1_EXP    
    L0_2 = 1.0 - A2_EXP
    L1_2 = 1.0 - (1.0 + A2D) * A2_EXP
    L2_2 = 1.0 - (1.0 + A2D + A2D_SQ) * A2_EXP
    A_A2_EXP, B_A1_EXP = A * A2_EXP, B * A1_EXP
    A1D_CU, A2D_CU = A1D_SQ * A1D, A2D_SQ * A2D   
    A1D_HY, A2D_HY = A1D_SQ * A1D_SQ, A2D_SQ * A2D_SQ
    L0 = lambda: 1 - A_A2_EXP - B_A1_EXP 
    L1 = lambda: 1 - (1 + A2D) * A_A2_EXP - (1 + A1D) * B_A1_EXP 
    L2 = lambda: 1 - (1 + A2D + THIRD * A2D_SQ) * A_A2_EXP\
                   - (1 + A1D + THIRD * A1D_SQ) * B_A1_EXP
    L3 = lambda: 1 - (1 + A2D + 0.4 * A2D_SQ + (1 / 15) * A2D_CU) * A_A2_EXP\
                   - (1 + A1D + 0.4 * A1D_SQ + (1 / 15) * A1D_CU) * B_A1_EXP
    L4 = lambda: 1 - (1 + A2D + (3 / 7) * A2D_SQ + (2 / 21) * A2D_CU + (1 / 105) * A2D_HY) * A_A2_EXP\
                   - (1 + A1D + (3 / 7) * A1D_SQ + (2 / 21) * A1D_CU + (1 / 105) * A1D_HY) * B_A1_EXP 
    L0_eq = lambda: 1 - (1 + 0.5 * A1D) * A1_EXP # l1
    L1_eq = lambda: 1 - (1 + A1D + 0.5 * A1D_SQ) * A1_EXP #l3
    L2_eq = lambda: 1 - (1 + A1D + 0.5 * A1D_SQ + (1 / 6) * A1D_CU) * A1_EXP
    L3_eq = lambda: 1 - (1 + A1D + 0.5 * A1D_SQ + (1 / 6) * A1D_CU + (1 / 30) * A1D_HY) * A1_EXP
    L4_eq = lambda: 1 - (1 + A1D + 0.5 * A1D_SQ + (1 / 6) * A1D_CU + (4 / 105) * A1D_HY + (1 / 210) * A1D_HY * A1D) * A1_EXP
    eq_cond = tf.math.abs(a_1 - a_2) < 1e-5    
    D0_pair = tf.where(eq_cond, L0_eq(), L0())    
    D1_pair = tf.where(eq_cond, L1_eq(), L1())
    D2_pair = tf.where(eq_cond, L2_eq(), L2())
    D3_pair = tf.where(eq_cond, L3_eq(), L3())
    D4_pair = tf.where(eq_cond, L4_eq(), L4())
    return L0_1, L1_1, L2_1, L0_2, L1_2, L2_2, D0_pair, D1_pair, D2_pair, D3_pair, D4_pair