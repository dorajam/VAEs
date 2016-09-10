# Based on the following tutorial: https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html

'''
Tensorflow is a programming system in which you represent computations as graphs where each node in the graph
is an opearation (op). Ops (may) take some tensors, carry out some computation, and spit out some new tensors.
Once this graph is built, a session (object)  is launched to execute the computation graph. Specifically, a session 
places the different parts of the graph onto different devices (e.g. GPUs, CPUs) and provides the instructions
on how to execute the computations. Your computations are distributed among your devices with a preference 
given to your GPU, if available.

'''

# LOAD MNIST DATA SET
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# (assemble phase:) BUILD GRAPH first then (execution phase:) START SESSION. 
# Else - best when using  IPython - you can start an INTERACTIVE SESSION to interleave the two.
import tensorflow as tf
sess = tf.InteractiveSession()

# To avoid the overhead from switching between doing computatins in other languages outside of Python (e.g. Numpy) 
# TF allows you to build an external graph in Python, then dictate which parts of the computation graph should be run.

