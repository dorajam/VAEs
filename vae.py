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
        






					
