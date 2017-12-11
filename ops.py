import tensorflow as tf
import tensorflow.contrib.slim as slim


def batch_norm(x, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def instance_norm(input, name='instance_norm'):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable('scale', [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


def conv2d(input_, output_dim, ks=3, s=1, stddev=0.02, padding='SAME', name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def relu(x, name='relu'):
    return tf.nn.relu(x)

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def tanh(x):
    return tf.nn.tanh(x)