from baselines.common.distributions import Pd, DiagGaussianPd
import numpy as np
import tensorflow as tf


class SquashedDiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = tf.tanh(mean)
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        _ = 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
            + tf.reduce_sum(self.logstd, axis=-1)
        _ += tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - self.sample() ** 2, l=0, u=1) + 1e-6), axis=1)
        return _

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (
                2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return tf.tanh(self.mean + self.std * tf.random_normal(tf.shape(self.mean)))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)
