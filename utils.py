import functools

from functional import compose, partial
import tensorflow as tf


def composeAll(*args):
	'''takes a list of functions and returns a composed function
	composed = composeAll([f,h,g])
	composed(x) = f(g(h(x)))'''

	return partial(functools.reduce, compose)(*args)

def print_(var, name: str, first_n=5, summarize=5):
    ''' print values of tf.Vars during training
    args: 
    input_ = tensor passed through op
    data = list of tensors to print out
    first_n = only the first n times
    summarize = only print this amount of entries in the tensor'''

    return tf.Print(var, [var], '{}'.format(name), first_n=first_n, summarize=summarize)
	
def get_mnist(n, mnist):
    ''' takes n (int in [0,9]), returns 784D numpy array '''
    assert 0 <= n <= 9, "n must be in [0-9]"
    import random
    SIZE = 500
    imgs, labels = mnist.train.next_batch(SIZE)
    imgs.shuffle()

    for i in xrange(len(imgs)):
        if labels[i] == n:
            return imgs[i]
