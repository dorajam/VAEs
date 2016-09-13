import functools

from functional import compose, partial
import tensorflow as tf


def composeAll(*args):
	'''takes a list of functions and returns a composed function
	composed = composeAll([f,h,g])
	composed(x) = f(g(h(x)))'''

	return partial(functools.reduce, compose)(*args)

	
