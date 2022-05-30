import tensorflow as tf

# Dispersion interaction based on Cn coefficients and the Becke-Johnson damping scheme proposed in
# https://aip.scitation.org/doi/10.1063/1.1949201
def BJ_dispersion(R2, C6, C8, C10):
    Rvdw = get_Rvdw(C6, C8, C10)
    Rvdw_2 = tf.square(Rvdw)
    Rvdw_6 = tf.math.pow(Rvdw_2, 3)
    Rvdw_8 = Rvdw_6 * Rvdw_2
    Rvdw_10 = Rvdw_8 * Rvdw_2
    R6 = tf.math.pow(R2, 3)
    R8 = R2 * R6
    R10 = R2 * R8    
    C6_term = C6 / (R6 + Rvdw_6)
    C8_term = C8 / (R8 + Rvdw_8)
    C10_term = C10 / (R10 + Rvdw_10)
    return -(C6_term + C8_term + C10_term)
    
# Combination rule for the BJ damping function as proposed in
# https://aip.scitation.org/doi/10.1063/1.2190220
def get_Rvdw(C6, C8, C10):
    return (tf.sqrt(C8 / C6) + tf.math.pow((C10 / C6), 0.25) + tf.sqrt(C10 / C8)) / 3

