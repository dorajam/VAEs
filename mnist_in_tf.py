# Based on the following tutorial: https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html

'''
Tensorflow is a programming system in which you represent computations as graphs where each node in the graph
is an opearation (op). Ops (may) take some tensors, carry out some computation, and spit out some new tensors.
Once this graph is built, a session (object)  is launched to execute the computation graph. Specifically, a session 
places the different parts of the graph onto different devices (e.g. GPUs, CPUs) and provides the instructions
on how to execute the computations. Your computations are distributed among your devices with a preference 
given to your GPU, if available.

'''

################################################################
# LOADING DATA
################################################################
# LOAD MNIST DATA SET
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


################################################################
# ASSEMBLE GRAPH
################################################################
# (assemble phase:) BUILD GRAPH first then (execution phase:) START SESSION. 
# Else - best when using  IPython - you can start an INTERACTIVE SESSION to interleave the two.
import tensorflow as tf
sess = tf.InteractiveSession()

# To avoid the overhead from switching between doing computatins in other languages outside of Python (e.g. Numpy) 
# TF allows you to build an external graph in Python, then dictate which parts of the computation graph should be run.

# Create nodes for holding the data
# both inputs are 2d tensors, None representing the batchsize
x = tf.placeholder(tf.float32, shape = [None, 784])
y_correct = tf.placeholder(tf.float32, shape= [None, 10])

# Create Variables for holding the current states of your parameters -> this allows you to change the state by any computations
#  through the graph as opposed to placeholders where the values cannot be changed
# initialize the variable to have values of zero
# before launching a session, you need to explicitly initialize ALL VARIABLES by: sess.run(tf.initialize_all_variables())
W = tf.Variable(tf.zeros([784,10])
b = tf.Variable(tf.zeros([10])

# REGRESSION MODEL
y = tf.nn.softmax(tf.matmul(x,W) + b)

# LOSS FUNCTION
# reduce_sum sums over all class scores -> while reduce_mean takes the average
cross_entropy = -tf.reduce_mean(-tf.reduce_sum(y_correct * tf.log(y), reduction_indices = [1]))

################################################################
# TRAINIG
################################################################
# learning rate/eta = 0.5
'''
 TF has built-in optimizers -> to create one with the desired parameters,
 you need to instantiate one of its subclasses. Then add teh optimizer to the ops 
 to minimize the cost by updating the parameters.
 e.g.
 opt = GradientDescentOptimizer(learning_rate=0.1)
 opt_op = opt.minimize(cost,var_list=< ALL tf.Variable>).
 opt_op.run()      # this does one step of training

 Calling minimize() both computes the gradients and applies them to the parameters (variables)
 You can also do this separately first calling compute_gradients(), then apply_gradients()
 '''
# don't forget to initialize all your varsj 
tf.initialize_all_variables().run()

# this combined all of the above and updates the params
# you can run this repeatedly to do all the training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
		batch = mnist.train.next_batch(100)
		train_step.run(feed_dict={x:batch[0], y:batch[1]})


################################################################
# EVALUATE 
################################################################
# tf.argmax is the same as np.argmax -> returns the index of the entry that's
#  equal to the num you specify

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_correct,1))      # returns a bool

# tf.cast = [True, False, True] -> [1,0,1] -> then take average
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# execure the eval funcs by applying them to the test data:
print(accuracy.eval(feed_dict={x:mnist.test.images, y_correct:mnist.test.labels}))


