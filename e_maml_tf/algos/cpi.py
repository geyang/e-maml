from collections import Sequence
import tensorflow as tf
from gym import spaces
from waterbear import Bear
from waterbear import OrderedBear

from e_maml_tf.config import G

# NOTE: best way to define the input interface is to use a named_tuple and then others could just import the tuple from
# here: https://pymotw.com/2/collections/namedtuple.html
# NOTE: However pickle has trouble with namedtuple. Plus a class offers more functions, so we use a namespace instead.
# InputT = namedtuple("Inputs", 'A ADV R OLD_NEG_LOG_P_AC OLD_V_PRED CLIP_RANGE X_act X_train')


# Here we use a input class to make it easy to define defaults.
from e_maml_tf.ge_utils import placeholders_from_variables


class Inputs:
    def __init__(self, *, action_space, value_baseline=False):
        if isinstance(action_space, spaces.Discrete):
            self.A = tf.placeholder(tf.int32, [None], name="A")
        else:
            self.A = tf.placeholder(tf.float32, [None] + list(action_space.shape), name="A")
        self.ADV = tf.placeholder(tf.float32, [None], name="ADV")
        self.OLD_NEG_LOG_P_AC = tf.placeholder(tf.float32, [None], name="OLD_NEG_LOG_P_AC")
        self.CLIP_RANGE = tf.placeholder(tf.float32, [], name="CLIP_RANGE")

        if value_baseline:
            self.R = tf.placeholder(tf.float32, [None], name="R")
            self.OLD_V_PRED = tf.placeholder(tf.float32, [None], name="OLD_V_PRED")


class Reports(OrderedBear):
    loss = None
    entropy = None
    approx_kl = None
    clip_frac = None


class CPI:
    vf_loss = None
    def __init__(self, *, inputs: Inputs, policy, vf_coef=None, ent_coef=None):
        self.inputs = inputs
        self.policy = policy
        with tf.variable_scope('CPI'):
            self.neglogpac = policy.pd.neglogp(inputs.A)
            entropy = tf.reduce_mean(policy.pd.entropy())

            ratio = tf.exp(inputs.OLD_NEG_LOG_P_AC - self.neglogpac)
            pg_loss = tf.reduce_mean(-inputs.ADV * ratio)
            self.loss = pg_loss - entropy * ent_coef

            if policy.vf is not None:
                assert vf_coef is not None, \
                    "vf_coef can not be None when policy has value function."
                vpred = policy.vf
                vf_loss = tf.square(vpred - inputs.R)
                self.loss += vf_loss * vf_coef

            self.reports = Reports(
                loss=self.loss,
                entropy=entropy,
                approx_kl=.5 * tf.reduce_mean(tf.square(self.neglogpac - inputs.OLD_NEG_LOG_P_AC)),
                clip_frac=tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), inputs.CLIP_RANGE)))
            )
            if policy.vf is not None:
                self.reports.vf_loss = vf_loss


class Optimize:
    optimize = None
    run_optimize = None

    def __init__(self, *, loss, trainables, lr=None, max_grad_norm=None, max_grad_clip=None,
                 strict=None, reports=None, **_):
        """
        :param trainables: Optional array used for the gradient calculation
        """
        with tf.variable_scope('CPI_Optimize'):
            # grad_placeholders = placeholders_from_variables(trainables)
            # optimizer.gradients is just a wrapper around tf.gradients, with extra assertions. This is why it raises
            # errors on non-trainables.
            _grads = tf.gradients(loss, trainables)
            if strict:
                for g in _grads:
                    assert g is not None, f'Some Grads are not defined: {_grads}'
            else:
                _grads = [tf.zeros_like(p) if g is None else g for g, p in zip(_grads, trainables)]

            assert (not max_grad_norm or not max_grad_clip), \
                f'max_grad_norm({max_grad_clip}) and max_grad_norm({max_grad_clip}) can not be trueful at the same time.'
            if max_grad_norm:  # allow 0 to be by-pass
                # print('setting max-grad-norm to', max_grad_norm)
                # tf.clip_by_global_norm is just fine. No need to use my own.
                _grads = [g * tf.stop_gradient(max_grad_norm / tf.maximum(max_grad_norm, tf.norm(g))) for g in _grads]
                # _grads, grad_norm = tf.clip_by_global_norm(_grads, max_grad_norm)
            elif max_grad_clip:
                _grads = [tf.clip_by_value(g, -max_grad_clip, max_grad_clip) for g in _grads]

            self.grads = _grads

            # graph operator for updating the parameter. used by maml with the SGD inner step
            self.apply_grad = lambda *, lr, grad, var: var - lr * grad

            if lr is not None:
                assert hasattr(trainables[0], '_variable'), "trainables have to have the _variable attribute"
                lr_not_scalar = (hasattr(lr, 'shape') and len(lr.shape)) or (isinstance(lr, Sequence) and len(lr))
                self.optimize = [v.assign(self.apply_grad(lr=lr[i] if lr_not_scalar else lr, grad=g, var=v))
                                 for i, (v, g) in enumerate(zip(trainables, self.grads))]
                _ = self.optimize if reports is None else [*vars(reports).values(), *self.optimize]
                self.run_optimize = lambda feed_dict: tf.get_default_session().run(_, feed_dict=feed_dict)

        # Function to compute the CPI gradients
        self.run_grads = lambda *, feed_dict: tf.get_default_session().run([_grads], feed_dict)


def path_to_feed_dict(*, inputs: Inputs, paths, lr=None, clip_range, **_r):
    if 'adv' in paths:
        phi = paths['advs']
    elif 'values' in paths:
        phi = paths['returns'] - paths['values']
    else:
        phi = paths['returns']
    # advs_normalized = (advs - advs.mean()) / (advs.std() + 1e-8)

    n_timesteps, n_envs, *_ = paths['obs'].shape
    n = n_timesteps * n_envs

    feed_dict = {
        inputs.X: paths['obs'].reshape(n, -1),
        inputs.A: paths['acs'].reshape(n, -1),
        inputs.ADV: phi.reshape(-1),
        inputs.OLD_NEG_LOG_P_AC: paths['neglogpacs'].reshape(-1),
        inputs.CLIP_RANGE: clip_range
    }
    if hasattr(inputs, 'OLD_V_PRED'):
        feed_dict[inputs.OLD_V_PRED] = paths['values'].reshape(-1)
        feed_dict[inputs.R] = paths['returns'].reshape(-1)
    if lr is not None:
        assert inputs.LR is not None, f'Input should have LR attribute if a learning rate is passed.'
        feed_dict[inputs.LR] = lr
    return feed_dict
