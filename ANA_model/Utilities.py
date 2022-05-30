import tensorflow as tf

S = lambda x, y: tf.reduce_sum(x * y, axis=-1, keepdims=True)
A = lambda x, k=-1: tf.expand_dims(x, axis=k)

def cdist_tf_batch(A, B):
    na = tf.reduce_sum(tf.square(A), axis=-1, keepdims=True)
    nb = tf.transpose(tf.reduce_sum(tf.square(B), axis=-1, keepdims=True), [0, 2, 1])
    return tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))

def D_Q(quadrupoles):
    return tf.linalg.set_diag(quadrupoles, tf.linalg.diag_part(quadrupoles) - tf.expand_dims((tf.linalg.trace(quadrupoles) / 3), axis=-1))

def outer_product(vectors):
    vectors = tf.expand_dims(vectors, axis=-1)
    return D_Q(vectors * tf.linalg.matrix_transpose(vectors))