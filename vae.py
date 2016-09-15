import os
import sys

import numpy as np
import tensorflow as tf

from layers import Dense
from utils import composeAll

class VAE(object):
	''' Variational Autoencoder'''

	# optional hyperparameters
	DEFAULTS = {
	    "batch_size":128,
		"learning_rate":1e-3,
		"dropout": 1.,
		"l2_lambda": 0.,
		"nonlinearity":tf.nn.sigmoid}
            
    RESTORE_KEY="to_restore"

	def __init__(self, architecture=[], d_hyperparams={}, meta_graph=None,
				save_graph=True, log_dir="./log"):
		''' Builds VAE model:
			
		* all args are keyword to keep track of what's what

		* architecure=[500,250,50] -> list of nodes for encoder
		final architecutre will be this mirroredi
		'''
			
		self.archtecture=architecture
		# if new hyperparams are given, update the default settings
		self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
		self.sesh = tf.Session()   # self.sesh.run(op) -> run specific op

		if not meta_graph:      # new model
            assert len(self.architecture) > 2, \
                "Layers must be at least: (input, hidden, latent)"
            
            # build graph
            handles = self._buildGraph()   # e.g. x_in, dropout, z_mean, z_log_sigma, x_reconstructed etc.
            for handle in handles:
                tf.add_to_collections(VAE.RESTORE_KEY, handle)       # remember the model structure

        else:  # restore
            model_name = os.path.basename(meta_graph).split("_vae_")


        # unpack handles for tensor ops to feed or fetch
        (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
         self.x_reconstructed, self.z_, self.x_reconstructed_,
         self.cost, self.global_step, self.train_op) = handles


    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, self.architecture[0]], name='x')
        # placeholder_with_default -> takes input, returns input if dropout not applied, else sparse tensor
        dropout = tf.placeholder_with_default(1., shape=[], name='dropout')

        # encoding layers
        encoding = [Dense("encoding", hidden_size, dropout, self.nonlinearity) for hidden_size in self.architecture[1:-1]]

        # spit out the encoded neurons before Z layer
        h_encoded = composeAll(reversed(encoding))(x_in)

        '''
        Here is the inference part: The hidden neurons on the last layer are dotted with the weights to generate the characterizing parameters of some random variable Z
        Z is an unobserved continuous variable which we'd like to extimate as accurately as possible.
        Step (i): generate z from some prior distribution p(z) - (Gaussian)
        Step (ii): generate x from p(x|z)
        We (initially randomly) generate these parameters after which we use our prior distribution (Gaussian) to draw a z value.

        '''
        # calculating z by matmuling the last encoded hidden neurons -> how neat is currying here
        # non-lineary = tf.identity
        # latentdistibution parameterized by hidden encoding
        # z ~ N(z_mean, np.exp(z_log_sigma)**2)
        z_mean = Dense("mean", self.architecture[-1], dropout)(h_encoded)
        z_log_sigma = Dense("z_log_sigma", self.architecutre[-1], dropout)(h_encoded)

        # adds epsilon random noise into the sampling process -> stochastic! 
        z = self.sampleGaussian(z_mean, z_log_sigma)

        decoding = [Dense("decoding", hidden_size, dropout, self.nonlinearity) 
                   for hidden_size in self.architecture[1:-1]]

        # resconstruct x
        decoding.insert(0, Dense('x_decoding', self.architecture[0], dropout, self.squashing))
        x_generated = tf.identity(composeAll(decoding)(z), name="x_generated")

        # directly explore latent space -> see decoding based on randomly generated z vals
        z_ = tf.placeholder_with_default(tf.random_normal([1, self.architecture[-1]]), # default mean=0.0, stddev=1.0
                                        shape=[None, self.architecture[-1]],
                                        name="latent_in")
        x_generated_ = composeAll(decoding)(z_)

        # reconstruction loss: difference b/w x and x_generated
        # binary cross_entropy - assumes x & p(x|z) are iid Bernoullis
        rec_loss = VAE.crossEntropy(x_generated, x_in)

        # mismatch b/w the extimated and true posterior distrib p(z|x)
        kl_loss = VAE.kullbackLeibler(z_mean, z_log_sigma) 
        




    def sampleGaussian(self, mu, log_sigma):
        ''' samples z from z~N(mu, log_sigma) with random noise epsilon
        this is necessary to make our inference and generation process differentiable'''
        with tf.name_scope('sample_gaussian'):
            epsilon = tf.random_normal(tf.shape(log_sigma), name='epsilon')
            return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)



					
    @staticmethod
    def crossEntropy(obs, actual, offset=1e-7):
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) +
                                  (1 - actual) * tf.log(1 - obs_), 1)


    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 -
                                        tf.exp(2 * log_sigma), 1)
