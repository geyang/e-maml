from collections import Sequence

import numpy as np
import tensorflow as tf
from gym import spaces
from waterbear import OrderedBear


# best way to define the input interface is to use a named_tuple and then others could just import the tuple from here:
# https://pymotw.com/2/collections/namedtuple.html
# InputT = namedtuple("Inputs", 'A ADV R OLD_NEG_LOG_P_AC OLD_V_PRED CLIP_RANGE X_act X_train')

# helper utilities
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        import numpy as np
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs1_buf[idxs],
                    obs_next=self.obs2_buf[idxs],
                    acs=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    dones=self.done_buf[idxs])


# Here we use a input class to make it easy to define defaults.
from e_maml_tf.ge_utils import placeholders_from_variables


class Inputs:
    def __init__(self, *, ob_shape, action_space, ):
        self.X = tf.placeholder(dtype=tf.float32, shape=ob_shape, name='obs')
        self.X_NEXT = tf.placeholder(dtype=tf.float32, shape=ob_shape, name='obs')

        if isinstance(action_space, spaces.Discrete):
            self.A = tf.placeholder(tf.int32, [None], name="A")
        else:
            self.A = tf.placeholder(tf.float32, [None] + list(action_space.shape), name="A")

        self.R = tf.placeholder(tf.float32, [None], name="R")
        self.DONE = tf.placeholder(tf.float32, [None], name="DONE")


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


class Critic:
    def __init__(self, inputs, pi, hidden_sizes=(400, 200), activation='relu', scope='Critic', reuse=False):
        """

        :param X: placehodler for th
        :param A: placeholder for the sampled actions
        :param pi: the reparameterized action
        :param hidden_sizes:
        :param activation:
        :param scope:
        :param reuse:
        """
        if activation == 'tanh':
            act = tf.tanh
        elif activation == "relu":
            act = tf.nn.relu
        else:
            raise TypeError(f"{activation} is not available in this MLP.")

        def vf_mlp(x):
            _ = mlp(x, [*hidden_sizes, 1], act, None)
            return tf.squeeze(_, 1)

        with tf.variable_scope(scope):
            # note: allow passing in trainables.
            _old_trainables = {*tf.trainable_variables()}

            x_a_ = tf.concat([inputs.X, inputs.A], -1)
            x_pi_ = tf.concat([inputs.X, pi], -1)
            with tf.variable_scope('Q_0', reuse=reuse):
                self.q_0 = vf_mlp(x_a_)
            with tf.variable_scope('Q_0', reuse=True):
                self.q_0_pi = vf_mlp(x_pi_)
            with tf.variable_scope('Q_1', reuse=reuse):
                self.q_1 = vf_mlp(x_a_)
            with tf.variable_scope('Q_1', reuse=True):
                self.q_1_pi = vf_mlp(x_pi_)

            _ = tf.trainable_variables()
            with tf.variable_scope('v', reuse=reuse):
                self.v = vf_mlp(inputs.X)
            self.v_trainables = [v for v in tf.trainable_variables() if v not in _]

            self.trainables = [v for v in tf.trainable_variables() if v not in _old_trainables]

            _ = tf.trainable_variables()
            with tf.variable_scope('v_target', reuse=reuse):
                self.v_targ = vf_mlp(inputs.X_NEXT)
            self.v_targ_trainables = [v for v in tf.trainable_variables() if v not in _]


class Reports(OrderedBear):
    value_loss = None
    pi_kl = None
    q0_loss = None
    q1_loss = None
    v_loss = None
    entropy = None
    act_norm = None


class SAC:
    from e_maml_tf.ge_policies import MlpPolicy
    def __init__(self, *, inputs: Inputs, policy: MlpPolicy, critic: Critic, polyak, ent_coef, gamma):
        self.inputs = inputs
        self.policy = policy
        self.critic = critic
        with tf.variable_scope('SAC'):
            min_q_pi = tf.minimum(critic.q_0_pi, critic.q_0_pi)
            q_backup = tf.stop_gradient(inputs.R + gamma * (1 - inputs.DONE) * critic.v_targ)
            v_backup = tf.stop_gradient(min_q_pi - ent_coef * policy.logpac)

            # this first term is using the Q function as an energy model to compute the KL divergence between
            # the policy distribution and the distribution from the Q function (critic)
            self.pi_loss = tf.reduce_mean(ent_coef * policy.logpac - critic.q_0_pi)
            q0_loss = 0.5 * tf.reduce_mean(tf.square(q_backup - critic.q_0))
            q1_loss = 0.5 * tf.reduce_mean(tf.square(q_backup - critic.q_1))
            v_loss = 0.5 * tf.reduce_mean(tf.square(v_backup - critic.v))
            self.value_loss = q0_loss + q1_loss + v_loss

            self.update_v_targ_ops = [tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                      for v_main, v_targ in zip(critic.v_trainables, critic.v_targ_trainables)]

            # entropy = tf.reduce_mean(policy.pd.entropy())
            self.reports = Reports(
                value_loss=self.value_loss,
                pi_kl=self.pi_loss,
                q0_loss=q0_loss,
                q1_loss=q1_loss,
                v_loss=v_loss,
                entropy=policy.entropy,
                act_norm=tf.reduce_mean(inputs.A),
            )


class Optimize:
    optimize = None
    run_optimize = None

    def __init__(self, policy_loss, policy_trainables, critic_loss, critic_trainables, lr=None, reports=None, **_):
        """
        :param trainables: Optional array used for the gradient calculation
        """
        with tf.variable_scope('SAC_Optimize'):
            # Note: optimizer.gradients is just a wrapper around tf.gradients, with extra assertions. This is why it
            #  raises errors on non-trainables.
            self.policy_grads = tf.gradients(policy_loss, policy_trainables)
            self.critic_grads = tf.gradients(critic_loss, critic_trainables)

            # graph operator for updating the parameter. used by maml with the SGD inner step
            self.apply_grad = lambda *, lr, grad, var: var - lr * grad

            if lr is not None:  # this is only called when we use this algo inside MAML, with SGD inner step.
                # todo: not used, not tested, but should be correct.
                assert hasattr(policy_trainables[0], '_variable'), "trainables have to have the _variable attribute"
                lr_not_scalar = (hasattr(lr, 'shape') and len(lr.shape)) or (isinstance(lr, Sequence) and len(lr))
                pi_opt_op = [v.assign(self.apply_grad(lr=lr[i] if lr_not_scalar else lr, grad=g, var=v))
                             for i, (v, g) in enumerate(zip(policy_trainables, self.policy_grads))]

                with tf.control_dependencies(pi_opt_op):
                    self.optimize = [v.assign(self.apply_grad(lr=lr[i] if lr_not_scalar else lr, grad=g, var=v))
                                     for i, (v, g) in enumerate(zip(critic_trainables, self.critic_grads))]

                _ = self.optimize if reports is None else [*vars(reports).values(), *self.optimize]
                self.run_optimize = lambda feed_dict: tf.get_default_session().run(_, feed_dict=feed_dict)


BUFFER: ReplayBuffer = None


def use_replay_buffer(buffer):
    global BUFFER
    BUFFER = buffer


def path_to_feed_dict(*, inputs: Inputs, paths, lr=None, **_r):
    """
    In SAC (and other value-based, non-policy gradient methods, where the policy gradient is provided
    by the true critic), the path_to_feed_dict function is stateful and contains a
    replay buffer.

    :param inputs:
    :param paths:
    :param lr:
    :param clip_range:
    :param _r:
    :return:
    """
    assert BUFFER is not None, "BUFFER is None. You need to first setup the replay buffer"

    buffer = BUFFER

    n_timesteps, n_envs, *_ = paths['obs'].shape
    n = n_timesteps * n_envs

    obs = paths['obs'].reshape(n, -1)
    acs = paths['acs'].reshape(n, -1)
    rewards = paths['rewards'].reshape(n, -1)
    dones = paths['dones'].reshape(n, -1)

    for step in range(1, n_timesteps):
        buffer.store(obs[step - 1], acs[step], rewards[step], obs[step], dones[step])

    _ = buffer.sample_batch(batch_size=n)

    feed_dict = {
        inputs.X: _['obs'],
        inputs.X_NEXT: _['obs_next'],
        inputs.A: _['acs'],
        inputs.R: _['rews'],
        inputs.DONE: _['dones']
    }
    if lr is not None:
        assert inputs.LR is not None, f'Input should have LR attribute if a learning rate is passed.'
        feed_dict[inputs.LR] = lr
    return feed_dict
