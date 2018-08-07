"""
This is the behavior cloning algorithm. Takes in observations and demonstration actions, to supervise-learn a
policy.

-- Ge

"""
import tensorflow as tf
from gym import spaces
from waterbear import OrderedBear


# NOTE: best way to define the input interface is to use a named_tuple and then others could just import the tuple from
# here: https://pymotw.com/2/collections/namedtuple.html
# NOTE: However pickle has trouble with namedtuple. Plus a class offers more functions, so we use a namespace instead.
# InputT = namedtuple("Inputs", 'A ADV R OLD_NEG_LOG_P_AC OLD_V_PRED CLIP_RANGE X_act X_train')


class Inputs:
    # note: we do not pass in the observation placeholder b/c it is not used at all in the code base.
    # note: this breaks consistency with the rest of the algos folder but we can fix it when see fit.
    def __init__(self, *, action_space, type=None):
        if type in [LOSS_TYPES.two_headed_BC, LOSS_TYPES.learned_loss_exp_act, LOSS_TYPES.learned_loss_deep,
                    LOSS_TYPES.learned_loss_exp_act_deep]:
            if isinstance(action_space, spaces.Discrete):
                self.A = tf.placeholder(tf.int32, [None], name="A")
            else:
                self.A = tf.placeholder(tf.float32, [None] + list(action_space.shape), name="A")


class Reports(OrderedBear):
    # note: does not include optional keys.
    loss = None
    act_norm = None
    entropy = None


class LOSS_TYPES:
    surrogate_target = "surrogate-target"
    a2_target = "a2_target"
    two_headed_BC = "two-headed-BC"
    learned_loss = "learned-BC-loss"
    learned_loss_deep = "learned-BC-loss-deep"  # this doesnt work as well as the action one.
    learned_loss_exp_act = "learned-BC-loss-with-expert-action"
    learned_loss_exp_act_deep = "learned-BC-loss-with-expert-action-deep"


def fc(x, scope, nh, act=tf.nn.relu):
    with tf.variable_scope(scope):
        nin = x.get_shape()[-1].value  # can take batched or individual tensors.
        w = tf.get_variable("w", [nin, nh], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h


class BCLearnedLoss:
    def __init__(self, *, inputs: Inputs, policy, type: str):
        self.inputs = inputs
        self.policy = policy
        with tf.variable_scope('BCLearnedLoss'):
            act_dim = policy.pd.mean.shape[-1]
            if type == LOSS_TYPES.surrogate_target:
                # learned loss. Use identity function as the activation
                surrogate_action = fc(policy.h_, 'surrogate_action_target', nh=act_dim, act=lambda x: x)
                self.loss = tf.reduce_mean(policy.pd.neglogp(surrogate_action))  # equivalent to L2 loss
                self.reports = Reports(
                    loss=self.loss,
                    act_norm=tf.reduce_mean(policy.pd.mean),
                    surrogate_act_norm=tf.reduce_mean(surrogate_action),
                    entropy=tf.reduce_mean(policy.pd.entropy())
                )
            elif type == LOSS_TYPES.a2_target:
                # two headed architecture. The policy head is not BC trained.
                action = fc(tf.concat([policy.h_, policy.pd.mean], -1), 'bc-surrogate-head', nh=act_dim,
                            act=lambda x: x)
                self.loss = tf.reduce_mean(action ** 2)
                self.reports = Reports(
                    loss=self.loss,
                    act_norm=tf.reduce_mean(policy.pd.mean),
                    surrogate_act_norm=tf.reduce_mean(action),
                    entropy=tf.reduce_mean(policy.pd.entropy())
                )
            elif type == LOSS_TYPES.two_headed_BC:
                # two headed architecture. The policy head is not BC trained.
                # Requires a BC action input
                surrogate_action = fc(tf.concat([policy.h_, policy.pd.mean], -1), 'surrogate_loss', nh=1,
                                      act=lambda x: x)
                self.loss = tf.reduce_mean(surrogate_action - inputs.A)
                self.reports = Reports(
                    loss=self.loss,
                    act_norm=tf.reduce_mean(policy.pd.mean),
                    entropy=tf.reduce_mean(policy.pd.entropy()),
                    surrogate_act_norm=tf.reduce_mean(surrogate_action),
                    expert_act_norm=tf.reduce_mean(inputs.A),
                )
            elif type == LOSS_TYPES.learned_loss:
                _ = tf.concat([inputs.X, policy.pd.mean], -1)
                _ = fc(_, 'learned_loss', nh=1, act=lambda x: x)
                self.loss = tf.reduce_mean(_ ** 2)  # effectively
                self.reports = Reports(
                    loss=self.loss,
                    act_norm=tf.reduce_mean(policy.pd.mean),
                    entropy=tf.reduce_mean(policy.pd.entropy())
                )
            elif type == LOSS_TYPES.learned_loss_deep:
                with tf.variable_scope('learned_loss'):
                    _ = tf.concat([inputs.X, policy.pd.mean], -1)
                    _ = fc(_, 'layer_1', nh=64)
                    _ = fc(_, 'layer_2', nh=1, act=lambda x: x)
                    self.loss = tf.reduce_mean(_ ** 2)  # effectively
                self.reports = Reports(
                    loss=self.loss,
                    act_norm=tf.reduce_mean(policy.pd.mean),
                    entropy=tf.reduce_mean(policy.pd.entropy())
                )
            elif type == LOSS_TYPES.learned_loss_exp_act:
                with tf.variable_scope('learned_loss'):
                    _ = tf.concat([inputs.X, inputs.A, policy.pd.mean], -1)
                    _ = fc(_, 'learned_loss', nh=1, act=lambda x: x)
                    self.loss = tf.reduce_mean(_ ** 2)  # effectively
                self.reports = Reports(
                    loss=self.loss,
                    expert_act_norm=tf.reduce_mean(inputs.A),
                    act_norm=tf.reduce_mean(policy.pd.mean),
                    entropy=tf.reduce_mean(policy.pd.entropy())
                )
            elif type == LOSS_TYPES.learned_loss_exp_act_deep:
                with tf.variable_scope('learned_loss'):
                    _ = tf.concat([inputs.X, inputs.A, policy.pd.mean], -1)
                    _ = fc(_, 'layer_1', nh=64)
                    _ = fc(_, 'layer_2', nh=1, act=lambda x: x)
                    self.loss = tf.reduce_mean(_ ** 2)  # effectively
                self.reports = Reports(
                    loss=self.loss,
                    expert_act_norm=tf.reduce_mean(inputs.A),
                    act_norm=tf.reduce_mean(policy.pd.mean),
                    entropy=tf.reduce_mean(policy.pd.entropy())
                )
            else:
                raise NotImplemented


# in behavior cloning, we use the supervising observation and actions.
# Assume the actions come from a gaussian policy
def path_to_feed_dict(*, inputs: Inputs, paths, lr=None, **_rest):
    """
    convert path objects to feed_dict for the tensorflow graph.

    :param inputs:  Input object
    :param paths: dict['obs', 'acs']: Size(n_timesteps, n_envs, feat_n)
    :param lr: placeholder or floating point number
    :param _rest:
    :return: feed_dict, keyed by the input placeholders.
    """
    # reshaping the path, need to debug
    n_timesteps, n_envs, *_ = paths['obs'].shape
    n = n_timesteps * n_envs

    feed_dict = {
        inputs.X: paths['obs'].reshape(n, -1),
    }
    if hasattr(inputs, 'A') and inputs.A is not None:
        feed_dict[inputs.A] = paths['acs'].reshape(n, -1)
    if lr is not None:
        assert inputs.LR is not None, f'Input should have LR attribute if a learning rate is passed.'
        feed_dict[inputs.LR] = lr
    return feed_dict
