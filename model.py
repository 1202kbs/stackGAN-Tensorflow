import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from ops import MNIST_Generator, MNIST_Discriminator
from Annotated_MNIST import Annotated_MNIST
from utils import ProgressBar, plot


class StackGAN(object):

    def __init__(self, config, sess):
        self.X0_dim = config.X0_dim
        self.X1_dim = config.X1_dim
        self.nwords = config.nwords
        self.vocab_size = config.vocab_size
        self.z_dim = config.z_dim
        self.c0_dim = config.c0_dim
        self.c1_dim = config.c1_dim
        self.e_dim = config.e_dim
        self.d_update = config.d_update
        self.g0_nepoch = config.g0_nepoch
        self.g1_nepoch = config.g1_nepoch
        self.batch_size = config.batch_size
        self.lmda = config.lmda
        self.lr = config.lr
        self.retrain_stage1 = config.retrain_stage1
        self.retrain_stage2 = config.retrain_stage2
        self.use_adam = config.use_adam
        self.show_progress = config.show_progress

        if self.use_adam:
            self.optimizer = tf.train.AdamOptimizer
        else:
            self.optimizer = tf.train.GradientDescentOptimizer

        self.checkpoint_dir = config.checkpoint_dir
        self.stage1_checkpoint_dir = os.path.join(config.checkpoint_dir, 'stage1')
        self.stage2_checkpoint_dir = os.path.join(config.checkpoint_dir, 'stage2')
        self.image_dir = config.image_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.stage1_checkpoint_dir)
            os.makedirs(self.stage2_checkpoint_dir)

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.annotator = Annotated_MNIST(train=True)

        self.X0 = tf.placeholder(tf.float32, [None, self.X0_dim], 'X0')
        self.X1 = tf.placeholder(tf.float32, [None, self.X1_dim], 'X1')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], 'z')
        self.context = tf.placeholder(tf.int32, [None, self.nwords], 'context')

        self.sess = sess


    def z_sampler(self, dim1, dim2):
        return np.random.uniform(-1, 1, size=[dim1, dim2])


    def build_model(self):
        E = tf.Variable(tf.random_normal([self.vocab_size, self.e_dim]))
        phi = tf.nn.embedding_lookup(E, self.context)
        phi = tf.reduce_sum(phi, axis=1)

        generator = MNIST_Generator(self.X0_dim, self.X1_dim, self.z_dim, self.c0_dim, self.c1_dim, self.e_dim)
        discriminator = MNIST_Discriminator(self.X0_dim, self.X1_dim, self.z_dim, self.c0_dim, self.c1_dim, self.e_dim)

        c0, self.G0 = generator.generate0(self.z, phi)
        c1, self.G1 = generator.generate1(self.G0, phi)

        self.D0_fake = discriminator.discriminate0(self.G0, phi, reuse=False)
        self.D0_real = discriminator.discriminate0(self.X0, phi, reuse=True)
        self.D1_fake = discriminator.discriminate1(self.G1, phi, reuse=False)
        self.D1_real = discriminator.discriminate1(self.X1, phi, reuse=True)

        gaussian_smp = tf.abs(tf.random_normal([]))

        self.G0_loss = -tf.reduce_mean(tf.log(self.D0_fake + 1e-10)) + self.lmda * tf.reduce_mean(c0 * tf.log(c0 / gaussian_smp + 1e-10))
        self.D0_loss = -tf.reduce_mean(tf.log(self.D0_real + 1e-10) + tf.log(1 - self.D0_fake + 1e-10))
        
        self.G1_loss = -tf.reduce_mean(tf.log(self.D1_fake + 1e-10)) + self.lmda * tf.reduce_mean(c1 * tf.log(c1 / gaussian_smp + 1e-10))
        self.D1_loss = -tf.reduce_mean(tf.log(self.D1_real + 1e-10) + tf.log(1 - self.D1_fake + 1e-10))
        
        G0_params = generator.get_vars('g0')
        D0_params = discriminator.get_vars('d0')
        G1_params = generator.get_vars('g1')
        D1_params = discriminator.get_vars('d1')

        G0_optimizer = self.optimizer(self.lr)
        self.G0_optim = G0_optimizer.minimize(loss=self.G0_loss, var_list=G0_params + [E])

        D0_optimizer = self.optimizer(self.lr)
        self.D0_optim = D0_optimizer.minimize(loss=self.D0_loss, var_list=D0_params + [E])

        G1_optimizer = self.optimizer(self.lr)
        self.G1_optim = G1_optimizer.minimize(loss=self.G1_loss, var_list=G1_params)

        D1_optimizer = self.optimizer(self.lr)
        self.D1_optim = D1_optimizer.minimize(loss=self.D1_loss, var_list=D1_params)

        tf.global_variables_initializer().run()
        self.saver1 = tf.train.Saver(var_list=G0_params + D0_params + [E])
        self.saver2 = tf.train.Saver(var_list=G1_params + D1_params)


    def train_stage1(self):
        avg_G0_loss = 0
        avg_D0_loss = 0
        iterations = int(self.annotator.batches.num_examples / self.batch_size)

        if self.show_progress:
            bar = ProgressBar('Train', max=iterations)

        for i in range(iterations):

            if self.show_progress:
                bar.next()

            descriptions, batch_xs, batch_xs_small, batch_ys = self.annotator.next_batch(self.batch_size, resize=True, convert_to_idx=True)

            feed_dict = {self.X0: batch_xs_small, self.context: descriptions, self.z: self.z_sampler(self.batch_size, self.z_dim)}

            for _ in range(self.d_update):
                _, D0_loss = self.sess.run([self.D0_optim, self.D0_loss], feed_dict=feed_dict)
            _, G0_loss = self.sess.run([self.G0_optim, self.G0_loss], feed_dict=feed_dict)
            
            avg_G0_loss += G0_loss / iterations
            avg_D0_loss += D0_loss / iterations

        if self.show_progress:
            bar.finish()

        return avg_G0_loss, avg_D0_loss


    def train_stage2(self):
        avg_G1_loss = 0
        avg_D1_loss = 0
        iterations = int(self.annotator.batches.num_examples / self.batch_size)

        if self.show_progress:
            bar = ProgressBar('Train', max=iterations)

        for i in range(iterations):

            if self.show_progress:
                bar.next()

            descriptions, batch_xs, batch_xs_small, batch_ys = self.annotator.next_batch(self.batch_size, resize=True, convert_to_idx=True)
            feed_dict = {self.X0: batch_xs_small, self.X1: batch_xs, self.context: descriptions, self.z: self.z_sampler(self.batch_size, self.z_dim)}

            for _ in range(self.d_update):
                _, D1_loss = self.sess.run([self.D1_optim, self.D1_loss], feed_dict=feed_dict)
            _, G1_loss = self.sess.run([self.G1_optim, self.G1_loss], feed_dict=feed_dict)
            
            avg_G1_loss += G1_loss / iterations
            avg_D1_loss += D1_loss / iterations

        if self.show_progress:
            bar.finish()

        return avg_G1_loss, avg_D1_loss


    def run(self):

        if not self.retrain_stage1 and not self.retrain_stage2:

            raise Exception('You must train at least one stage')

        if self.retrain_stage1:

            for epoch in range(self.g0_nepoch):
                avg_G0_loss, avg_D0_loss = self.train_stage1()

                state = {'G0 Loss': avg_G0_loss, 'D0 Loss': avg_D0_loss, 'Epoch': epoch}
                print(state)

                if epoch % 5 == 0:
                    context = self.annotator.generate_sentences(16, divide=True, convert_to_idx=True)
                    samples = self.sess.run(self.G0, feed_dict={self.context: context, self.z: self.z_sampler(16, self.z_dim)})
                    fig = plot(samples, [14, 14])
                    plt.savefig(os.path.join(self.image_dir, 'Stage1_{:04d}.png'.format(epoch)), bbox_inches='tight')
                    plt.close(fig)

                    self.saver1.save(self.sess, os.path.join(self.stage1_checkpoint_dir, 'StackGAN.model'))
        
        else:

            self.load(load_stage1=True, load_stage2=False)

        if self.retrain_stage2:

            for epoch in range(self.g1_nepoch):
                avg_G1_loss, avg_D1_loss = self.train_stage2()

                state = {'G1 Loss': avg_G1_loss, 'D1 Loss': avg_D1_loss, 'Epoch': epoch}
                print(state)

                if epoch % 5 == 0:
                    context = self.annotator.generate_sentences(16, divide=True, convert_to_idx=True)

                    samples_s = self.sess.run(self.G0, feed_dict={self.context: context, self.z: self.z_sampler(16, self.z_dim)})
                    fig = plot(samples_s, [14, 14])
                    plt.savefig(os.path.join(self.image_dir, 'Stage2_{:04d}_small.png'.format(epoch)), bbox_inches='tight')
                    plt.close(fig)

                    samples = self.sess.run(self.G1, feed_dict={self.context: context, self.z: self.z_sampler(16, self.z_dim)})
                    fig = plot(samples, [28, 28])
                    plt.savefig(os.path.join(self.image_dir, 'Stage2_{:04d}_large.png'.format(epoch)), bbox_inches='tight')
                    plt.close(fig)

                    self.saver2.save(self.sess, os.path.join(self.stage2_checkpoint_dir, 'StackGAN.model'))

        else:

            self.load(load_stage1=False, load_stage2=True)


    def generate(self, sentences):
        self.load(load_stage1=True, load_stage2=True)

        num_instances = len(sentences)
        context = self.annotator.convert_to_idx(sentences)
        return self.sess.run([self.G0, self.G1], feed_dict={self.context: context, self.z: self.z_sampler(num_instances, self.z_dim)})


    def load(self, load_stage1=True, load_stage2=True):

        if load_stage1:

            print('[*] Reading Stage 1 Checkpoints...')

            ckpt = tf.train.get_checkpoint_state(self.stage1_checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver1.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise Exception('[!] No Checkpoints Found')

        if load_stage2:

            print('[*] Reading Stage 2 Checkpoints...')

            ckpt = tf.train.get_checkpoint_state(self.stage2_checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver2.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise Exception('[!] No Checkpoints Found')
