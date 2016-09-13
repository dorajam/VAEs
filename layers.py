# Creates Dense layer obj; this will be used to build the model
import tensorflow as tf

class Dense(object):
	def __init__(self, scope="dense_layer", size=None, dropout=1., 
				nonlinearity=tf.identity):
		assert size, "Must specify layer size"
		self.scope = scope
		self.size = size
		self.dropout = dropout
		self.nonlinearity = nonlinearity

	def __call__(self, x):
		'''Currying: Treat Dense() obj as a method:
		calling Dense(<init args>)(x_input) --> does all matrix ops'''
		with tf.name_scope(self.scope):
			while True:
				try:
					return self.nonlinearity(tf.matmul(x, self.w) + self.b)

				# if self.w * self.b not yet initialized
				except(AttributeError):			
					self.w, self.b = self.init_parameters(x.get_shape()[1].value, self.size)
					self.w = tf.nn.dropout(self.w, self.dropout)


	@staticmethod
	def init_parameters(dim_in, dim_out):
		
		# casting will turn the floats into tf.float32
		stddev = tf.cast((2/dim_in)**0.5, tf.float32)    # Xavier init 1/n_in * 2 due to RelU's 0 outputs
		
		initial_w = tf.random_normal(shape=[dim_in, dim_out], stddev = stddev)
		initial_b = tf.zeros([dim_out])
		
		# trainable = True adds the vars to the list of trainable vars-> global key for GraphKeys
		return (tf.Variable(initial_w, trainable=True, name="weights"),
				tf.Variable(initial_b, trainable=True, name="biases"))
		


