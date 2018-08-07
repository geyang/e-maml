"""
This is the behavior cloning algorithm. Takes in observations and demonstration actions, to supervise-learn a
policy.

-- Ge

"""
from collections import defaultdict, Sequence
import tensorflow as tf
from gym import spaces
from typing import Callable, Union
from waterbear import OrderedBear

# NOTE: best way to define the input interface is to use a named_tuple and then others could just import the tuple from
# here: https://pymotw.com/2/collections/namedtuple.html
# NOTE: However pickle has trouble with namedtuple. Plus a class offers more functions, so we use a namespace instead.
# InputT = namedtuple("Inputs", 'A ADV R OLD_NEG_LOG_P_AC OLD_V_PRED CLIP_RANGE X_act X_train')


# Here we use a input class to make it easy to define defaults.
from e_maml_tf.ge_utils import placeholders_from_variables


class Inputs:
    def __init__(self, *, action_space, A=None):
        if isinstance(action_space, spaces.Discrete):
            self.A = A or tf.placeholder(tf.int32, [None], name="A")
        else:
            self.A = A or tf.placeholder(tf.float32, [None] + list(action_space.shape), name="A")


class Reports(OrderedBear):
    loss = None
    act_norm = None
    targ_act_norm = None
    entropy = None


class BC:
    def __init__(self, *, inputs: Inputs, policy):
        self.inputs = inputs
        self.policy = policy
        with tf.variable_scope('BC'):
            self.loss = tf.reduce_mean(policy.pd.neglogp(inputs.A))  # equivalent to L2 loss
            self.reports = Reports(
                loss=self.loss,
                act_norm=tf.reduce_mean(policy.pd.mean),
                targ_act_norm=tf.reduce_mean(inputs.A),
                entropy=tf.reduce_mean(policy.pd.entropy())
            )


class Optimize:
    optimize = None
    run_optimize = None

    def __init__(self, *, loss, trainables, lr=None, max_grad_norm=None, max_grad_clip=None, strict=False,
                 reports=None, **_):
        """
        Graph constructor for the optmizer

        :param lr: The learning rate, usually a placeholder but can be a float. Not needed if using external optimizer,
                    Needed here for the SGD update in the inner-step.
                    If set to None, then does not construct the self.optimize operator and the self.run_optimize
                    function.
        :param loss:
        :param trainables: Optional array used for the gradient calculation
        :param max_grad_norm:
        :param optimizer:
        :param _:
        """
        with tf.variable_scope('BC_Optimize'):
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


# note: this is singleton. Doesn't support having two instances. Move to class if that is needed.
# sampling helpers for demonstration data
SAMPLE_GENS = defaultdict(lambda: None)


def sample_generator(*, paths_list: Union[list], batch_size=None, augment_fn: Union[None, Callable] = None,
                     episodic_subsample_interval=1):
    """
    mode == "timestep":
        The sampler samples each timestep individually. Different rollouts are always sampled individually.

    mode == "episode":
        Each episode (index = 1) are sampled individually (rollout). Timesteps are not shuffled.

    Episodic Subsampling:
        Only applies under mode == "episode".
        
        The episodic subsample occurs at fixed interval. The starting point of this subsampling is randomly sampled.


    :param paths: dict['obs', 'acs'], values are tensors of the shape

                Size(timesteps, n_envs, feat_n).

            This makes it easier to manipulate shuffling and slicing timestep wise.

    :param batch_size: size for the mini-batches.
    :param augment_fn:  A function (*, obs, acs, *task_spec) => augmented path{obs, acs}
            Note: This augment_fn is called every mini-batch. It is task-specific. (takes in task_spec)
    :return: dict(
                 obs = Size(batch_size, 1, feat_n),
                 acs = Size(batch_size, 1, feat_n)
                 ...
             )
    """
    import numpy as np

    # assert mode is 'multitask', "Only multitask mode is supported now."
    # assert augment_fn is None, "The augmentation function is not called under this mode."
    # Now allow augment_fn in multitask mode.

    p0 = paths_list[0]  # assume that all data are identical shape.
    assert p0['obs'].shape[0] == p0['acs'].shape[0], "observation and actions need to have the same length."
    assert len(p0['obs'].shape) == 3, "observation (and action) are rank 3 tensors ~ Size(k, horizon, feat_n)."

    timesteps, k_rollouts, _ = p0['obs'].shape
    batch_size = batch_size or timesteps
    batch_n = timesteps * k_rollouts // batch_size
    assert timesteps % (episodic_subsample_interval * batch_size) == 0, f's.t. that shuffling works. ' \
        f'{timesteps} % ({episodic_subsample_interval} * {batch_size}) != 0'

    # the first next returns the number of batch :)
    task_spec = yield dict(batch_n=batch_n)

    assert timesteps % episodic_subsample_interval == 0, "has to be the right shape"
    new_shape = [episodic_subsample_interval, timesteps // episodic_subsample_interval, k_rollouts, -1]
    final_shape = [k_rollouts * episodic_subsample_interval, timesteps // episodic_subsample_interval, -1]
    paths = [{k: v.reshape(new_shape).swapaxes(1, 2).reshape(final_shape) if hasattr(v, 'shape') else v
              for k, v in _.items()} for _ in paths_list]
    while True:
        shuffled_inds = np.random.rand(episodic_subsample_interval * k_rollouts).argsort()
        # do all of the copying here.
        shuffled_paths = [{
            k: v[shuffled_inds].reshape(timesteps * k_rollouts, -1) if isinstance(v, np.ndarray)
            else v for k, v in _.items()} for _ in paths]
        for i in range(batch_n):
            task_index = task_spec['index'] if task_spec else 0
            selected_paths = shuffled_paths[task_index]

            start = i * batch_size
            # no copy involved
            batch_paths = {
                k: v[start: start + batch_size].reshape(batch_size, 1, -1) if isinstance(v, np.ndarray)
                else v for k, v in selected_paths.items()
            }
            # obs_augment occurs here
            # note: pass in index=task_index explicitly, b/c task_spec can be None.
            task_spec = yield augment_fn(**batch_paths, index=task_index) if augment_fn else batch_paths


def use_samples(key=None, **kwargs):
    global SAMPLE_GENS
    key = 'default' if key is None else key
    SAMPLE_GENS[key] = sample_generator(**kwargs)
    return next(SAMPLE_GENS[key])  # start the generator, return information of the generator.


DATA_MODE = "multi-mode"  # OneOf['multi-mode', 'simple']


# key = None or key = eval
def sample_demonstration_data(task_spec=None, key=None):
    global SAMPLE_GENS
    import numpy as np
    # add logic here to support multi-mode.
    if DATA_MODE == "multi-mode":
        if key is None:
            keys = [k for k in SAMPLE_GENS.keys() if "/" not in k]
        else:
            keys = [k for k in SAMPLE_GENS.keys() if k == key or k.startswith(key + "/")]
        key = keys[np.random.choice(len(keys))]
    elif DATA_MODE == "simple":
        key = 'default' if key is None else key
    else:
        raise NotImplementedError

    g = SAMPLE_GENS.get(key, None)
    assert g is not None, f'sample key {key} does NOT exist. First call use_samples to setup this sample gen.'
    return next(g) if task_spec is None else g.send(task_spec)


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
        inputs.A: paths['acs'].reshape(n, -1),
        # all of these are gone.
        # inputs.OLD_NEG_LOG_P_AC: paths['neglogpacs'].reshape(-1),
        # inputs.OLD_V_PRED: paths['values'].reshape(-1),
        # These are useful if the agent receives the reward.
        # inputs.ADV: advs_normalized.reshape(-1),
        # inputs.R: paths['returns'].reshape(-1),
        # inputs.CLIP_RANGE: clip_range
    }
    if lr is not None:
        assert inputs.LR is not None, f'Input should have LR attribute if a learning rate is passed.'
        feed_dict[inputs.LR] = lr
    return feed_dict
