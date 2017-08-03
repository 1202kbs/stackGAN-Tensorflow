import tensorflow as tf
import tensorflow.contrib.layers as tcl


class MNIST_Generator(object):

    def __init__(self, X0_dim, X1_dim, z_dim, c0_dim, c1_dim, e_dim):
        self.X0_dim = X0_dim
        self.X1_dim = X1_dim
        self.z_dim = z_dim
        self.c0_dim = c0_dim
        self.c1_dim = c1_dim
        self.e_dim = e_dim


    def generate0(self, z, phi):

        with tf.variable_scope('g0') as scope:
            G0_W1 = tf.get_variable('G0_W1', [self.e_dim, self.c0_dim], initializer=tcl.xavier_initializer())
            G0_b1 = tf.get_variable('G0_b1', [self.c0_dim], initializer=tf.constant_initializer())
            G0_W2 = tf.get_variable('G0_W2', [self.z_dim + self.c0_dim, self.X0_dim], initializer=tcl.xavier_initializer())
            G0_b2 = tf.get_variable('G0_b2', [self.X0_dim], initializer=tf.constant_initializer())

            c = tf.nn.sigmoid(tf.matmul(phi, G0_W1) + G0_b1)
            code = tf.concat([z, c], axis=1)

            layer1 = tf.nn.sigmoid(tf.matmul(code, G0_W2) + G0_b2)

        return c, layer1


    def generate1(self, X0, phi):

        with tf.variable_scope('g1') as scope:
            G1_W1 = tf.get_variable('G1_W1', [self.e_dim, self.c1_dim], initializer=tcl.xavier_initializer())
            G1_b1 = tf.get_variable('G1_b1', [self.c1_dim], initializer=tf.constant_initializer())
            G1_W2 = tf.get_variable('G1_W2', [self.X0_dim + self.c1_dim, self.X1_dim], initializer=tcl.xavier_initializer())
            G1_b2 = tf.get_variable('G1_b2', [self.X1_dim], initializer=tf.constant_initializer())

            c = tf.nn.sigmoid(tf.matmul(phi, G1_W1) + G1_b1)
            code = tf.concat([X0, c], axis=1)

            layer1 = tf.nn.sigmoid(tf.matmul(code, G1_W2) + G1_b2)

        return c, layer1


    def get_vars(self, scope_name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


class MNIST_Discriminator(object):

    def __init__(self, X0_dim, X1_dim, z_dim, c0_dim, c1_dim, e_dim):
        self.X0_dim = X0_dim
        self.X1_dim = X1_dim
        self.z_dim = z_dim
        self.c0_dim = c0_dim
        self.c1_dim = c1_dim
        self.e_dim = e_dim


    def discriminate0(self, X0, phi, reuse=False):

        with tf.variable_scope('d0') as scope:

            if reuse:
                scope.reuse_variables()

            D0_W1 = tf.get_variable('D0_W1', [self.e_dim, self.c0_dim], initializer=tcl.xavier_initializer())
            D0_b1 = tf.get_variable('D0_b1', [self.c0_dim], initializer=tf.constant_initializer())
            D0_W2 = tf.get_variable('D0_W2', [self.X0_dim, self.z_dim], initializer=tcl.xavier_initializer())
            D0_b2 = tf.get_variable('D0_b2', [self.z_dim], initializer=tf.constant_initializer())
            D0_W3 = tf.get_variable('D0_W3', [self.z_dim + self.c0_dim, 1], initializer=tcl.xavier_initializer())
            D0_b3 = tf.get_variable('D0_b3', [1], initializer=tf.constant_initializer())

            c = tf.nn.sigmoid(tf.matmul(phi, D0_W1) + D0_b1)
            z = tf.nn.sigmoid(tf.matmul(X0, D0_W2) + D0_b2)
            code = tf.concat([z, c], axis=1)

            layer1 = tf.nn.sigmoid(tf.matmul(code, D0_W3) + D0_b3)

        return layer1


    def discriminate1(self, X1, phi, reuse=False):

        with tf.variable_scope('d1') as scope:

            if reuse:
                scope.reuse_variables()

            D1_W1 = tf.get_variable('D1_W1', [self.e_dim, self.c1_dim], initializer=tcl.xavier_initializer())
            D1_b1 = tf.get_variable('D1_b1', [self.c1_dim], initializer=tf.constant_initializer())
            D1_W2 = tf.get_variable('D1_W2', [self.X1_dim, 128], initializer=tcl.xavier_initializer())
            D1_b2 = tf.get_variable('D1_b2', [128], initializer=tf.constant_initializer())
            D1_W3 = tf.get_variable('D1_W3', [128 + self.c1_dim, 1], initializer=tcl.xavier_initializer())
            D1_b3 = tf.get_variable('D1_b3', [1], initializer=tf.constant_initializer())

            c = tf.nn.sigmoid(tf.matmul(phi, D1_W1) + D1_b1)
            z = tf.nn.sigmoid(tf.matmul(X1, D1_W2) + D1_b2)
            code = tf.concat([z, c], axis=1)

            layer1 = tf.nn.sigmoid(tf.matmul(code, D1_W3) + D1_b3)

        return layer1

    def get_vars(self, scope_name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)

