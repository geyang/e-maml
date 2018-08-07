import numpy as np
import tensorflow as tf
from gym import spaces

from baselines.a2c.utils import ortho_init
from baselines.common.distributions import make_pdtype


def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[-1].value  # can take batched or individual tensors.
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h


LOG_STD_MAX = 10
LOG_STD_MIN = -10


class MlpPolicy:
    vf = None

    def __repr__(self):
        return f"{self.__class__} {self.name}"

    # noinspection PyPep8Naming
    def __init__(self, ac_space, X, hidden_size, n_layers=2, activation="tanh", value_baseline=False,
                 scope='MlpPolicy', reuse=False, X_placeholder=None, fix_variance=False, init_logstd=None):
        """
        Gaussian Policy. The variance is learned as parameters. You can also pass in the logstd from the outside.

            __init__: Construct the graph for the MLP policy.

        :param ac_space: action space, one of `gym.spaces.Box`
        :param X: Tensor or input placeholder for the observation
            :param hidden_size: size of hidden layers in network
        :param activation: one of 'reLU', 'tanh'
        :param scope: str, name of variable scope.
        :param reuse:
        :param value_baseline: bool flag whether compute a value baseline
        :param X_placeholder:
        :param fix_variance:
        :param init_logstd:
        """
        assert n_layers >= 2, f"hey, what's going on with this puny {n_layers}-layer network? " \
            f"--Ge (your friendly lab-mate)"
        if isinstance(scope, tf.VariableScope):
            self.scope_name = scope.name
        else:
            self.scope_name = scope
        self.name = (self.scope_name + "_reuse") if reuse else self.scope_name

        self.X_ph = X if X_placeholder is None else X_placeholder

        # done: this only applies to Discrete action space. Need to make more general.
        # now it works for both discrete action and gaussian policies.
        if isinstance(ac_space, spaces.Discrete):
            act_dim = ac_space.n
        else:
            act_dim, *_ = ac_space.shape

        if activation == 'tanh':
            act = tf.tanh
        elif activation == "relu":
            act = tf.nn.relu
        else:
            raise TypeError(f"{activation} is not available in this MLP.")
        with tf.variable_scope(scope, reuse=reuse):
            h_ = X
            for i in range(1, n_layers + 1):  # there is no off-by-one error here --Ge.
                h_ = fc(h_, f'pi_fc_{i}', nh=hidden_size, init_scale=np.sqrt(2), act=act)
                # a_ = fc(h_, f'pi_attn_{i}', nh=h_.shape[1], init_scale=np.sqrt(2), act=tf.math.sigmoid)
                # h_ = fc(h_ * a_, f'pi_fc_{i}', nh=hidden_size, init_scale=np.sqrt(2), act=act)
            mu = fc(h_, 'pi', act_dim, act=lambda x: x, init_scale=0.01)
            # _ = fc(h2, 'pi', act_dim, act=tf.tanh, init_scale=0.01)
            # mu = ac_space.low + 0.5 * (ac_space.high - ac_space.low) * (_ + 1)

            self.h_ = h_  # used for learned loss

            # assert (not G.vf_coef) ^ (G.baseline == "critic"), "These two can not be true or false at the same time."
            if value_baseline:
                # todo: conditionally declare these only when used
                # h1 = fc(X, 'vf_fc1', nh=hidden_size, init_scale=np.sqrt(2), act=act)
                # h2 = fc(h1, 'vf_fc2', nh=hidden_size, init_scale=np.sqrt(2), act=act)
                self.vf = fc(self.h_, 'vf', 1, act=lambda x: x)[:, 0]

            if isinstance(ac_space, spaces.Box):  # gaussian policy requires logstd
                shape = tf.shape(mu)[0]
                if fix_variance:
                    _ = tf.ones(shape=[1, act_dim], name="unit_logstd") * (init_logstd or 0)
                    logstd = tf.tile(_, [shape, 1])
                elif init_logstd is not None:
                    _ = tf.get_variable(name="logstd", shape=[1, act_dim],
                                        initializer=tf.constant_initializer(init_logstd))
                    # todo: clip logstd to limit the range.
                    logstd = tf.tile(_, [shape, 1])
                else:
                    # use variance network when no initial logstd is given.
                    # _ = fc(X, 'logstd_fc1', nh=hidden_size, init_scale=np.sqrt(2), act=act)
                    # _ = fc(_, 'logstd_fc2', nh=hidden_size, init_scale=np.sqrt(2), act=act)

                    # note: this doesn't work. Really need to bound the variance.
                    # logstd = 1 + fc(self.h_, 'logstd', act_dim, act=lambda x: x, init_scale=0.01)
                    logstd = fc(self.h_, 'logstd', act_dim, act=lambda x: x, init_scale=0.01)
                    # logstd = fc(self.h2, 'logstd', act_dim, act=tf.tanh, init_scale=0.01)
                    # logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)

                # GaussianPd takes 2 * [act_length] b/c of the logstd concatenation.
                ac = tf.concat([mu, logstd], axis=1)
                # A much simpler way is to multiply _logstd with a zero tensor shaped as mu.
                # [mu, mu * 0 + _logstd]
            else:
                raise NotImplemented('Discrete action space is not implemented!')

            # list of parameters is fixed at graph time.
            # todo: Only gets trainables that are newly created by the current policy function.
            # self.trainables = tf.trainable_variables()

            # placeholders = placeholders_from_variables(self.trainables)
            # self._assign_placeholder_dict = {t.name: p for t, p in zip(self.trainables, placeholders)}
            # self._assign_op = tf.group(*[v.assign(p) for v, p in zip(self.trainables, placeholders)])

        with tf.variable_scope("Gaussian_Action"):
            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(ac)

            self.a = a = self.pd.sample()
            self.mu = self.pd.mode()
            self.neglogpac = self.pd.neglogp(a)

    @property
    def trainables(self):
        raise DeprecationWarning("deprecated b/c bias transform.")

    @property
    def state_dict(self):
        # todo: should make the tensor names scoped locally.
        return {t.name: v for t, v in zip(self.trainables, tf.get_default_session().run(self.trainables))}

    # def load_from_state_dict(self, state_dict):
    #     # todo: this adds new assign ops each time, and causes the graph to grow.
    #     feed_dict = {self._assign_placeholder_dict[t.name]: state_dict[t.name] for t in self.trainables}
    #     return tf.get_default_session().run(self._assign_op, feed_dict=feed_dict)

    def step(self, ob, soft, feed_dict=None):
        if feed_dict:
            feed_dict.update({self.X_ph: ob})
        else:
            feed_dict = {self.X_ph: ob}
        sess = tf.get_default_session()
        if self.vf is None:
            ts = [self.a if soft else self.mu, self.neglogpac]
            return sess.run(ts, feed_dict=feed_dict)
        else:
            ts = [self.a if soft else self.mu, self.vf, self.neglogpac]
            return sess.run(ts, feed_dict=feed_dict)

    act = step

    def value(self, ob, feed_dict=None):
        if feed_dict:
            feed_dict.update({self.X_ph: ob})
        else:
            feed_dict = {self.X_ph: ob}
        sess = tf.get_default_session()
        return sess.run(self.vf, feed_dict=feed_dict)
