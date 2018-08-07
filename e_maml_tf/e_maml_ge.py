"""stands for 'correction-MAML. Could also argue complete-maml. Whatever."""
from collections import Sequence
from numbers import Number
from typing import Any

import matplotlib
from tqdm import trange
import tensorflow as tf

from e_maml_tf.ge_utils import get_scope_name, stem
from .config import G
from e_maml_tf.algos.vpg import Inputs as VPGInputs, VPG, Optimize as VPG_Optimize
from e_maml_tf.algos.ppo2 import Inputs as PPOInputs, PPO, Optimize as PPO_Optimize
from e_maml_tf.algos.cpi import Inputs as CPIInputs, CPI, Optimize as CPI_Optimize
from e_maml_tf.algos.bc import Inputs as BCInputs, BC, Optimize as BC_Optimize
from e_maml_tf.algos.bc_learned_loss import Inputs as BCLearnedLossInputs, BCLearnedLoss
from .ge_utils import defaultlist, make_with_custom_variables, GradientSum, Cache, var_map

matplotlib.use("Agg")

import baselines.common.tf_util as U
from .ge_policies import MlpPolicy

ALLOWED_ALGS = ('VPG', 'PPO', 'CPI', "BC", "BCLearnedLoss")


class Meta:
    optim = None

    def __init__(self, *, scope_name, act_space, ob_shape, algo, reuse: Any = False, trainables=None, optimizer=None,
                 add_loss=None, loss_only=False, lr_rank=None, max_grad_norm=None, max_grad_clip=None,
                 fix_variance=False):
        """
        Meta Graph Constructor

        :param scope_name:
        :param act_space:
        :param ob_shape:
        :param algo:
        :param reuse:
        :param trainables:
        :param optimizer:
        :param lr_rank: One of [None, 0, 1, 2] corresponding to [(), 'scalar', 'simple', "full"] learned learning rate.
        :param max_grad_norm:
        :param max_grad_clip:
        :param fix_variance:
        """
        assert algo in ALLOWED_ALGS, "model algorithm need to be one of {}".format(ALLOWED_ALGS)
        with tf.variable_scope(scope_name, reuse=reuse):
            obs = tf.placeholder(dtype=tf.float32, shape=ob_shape, name='obs')  # obs
            if algo == "PPO":
                self.inputs = inputs = PPOInputs(action_space=act_space, value_baseline=(G.baseline == "critic"))
                Optimize = PPO_Optimize
            elif algo == "VPG":
                self.inputs = inputs = VPGInputs(action_space=act_space, value_baseline=(G.baseline == "critic"))
                Optimize = VPG_Optimize
            elif algo == "CPI":
                self.inputs = inputs = CPIInputs(action_space=act_space, value_baseline=(G.baseline == "critic"))
                Optimize = CPI_Optimize
            elif algo == "BC":
                self.inputs = inputs = BCInputs(action_space=act_space)
                Optimize = BC_Optimize
            elif algo == "BCLearnedLoss":
                self.inputs = inputs = BCLearnedLossInputs(action_space=act_space, type=G.learned_loss_type)
                Optimize = BC_Optimize
            else:
                raise NotImplementedError(
                    'Only supports PPO, VPG, CPI, BC and BC with Learned Loss (BCLearnedLoss)')
            inputs.X = obs  # https://github.com/tianheyu927/mil/blob/master/mil.py#L218
            bias_transformation = tf.get_variable('input_bias', [1, G.bias_dim], initializer=tf.zeros_initializer())
            batch_n = tf.shape(obs)[0]
            trans_input = tf.tile(bias_transformation, [batch_n, 1])
            self.policy = policy = MlpPolicy(
                ac_space=act_space, hidden_size=G.hidden_size, n_layers=G.n_layers,
                activation=G.activation, value_baseline=(G.baseline == "critic"),
                reuse=reuse, X=tf.concat(values=(obs, trans_input), axis=1), X_placeholder=obs,
                init_logstd=G.init_logstd, fix_variance=fix_variance)

            # note that policy.trainables are the original trainable parameters, not the mocked variables.
            # todo: concatenate policy.trainable with local trainable (bias_transformation)
            self.trainables = tf.trainable_variables() if trainables is None else trainables

            ext_loss = add_loss(inputs.ADV) if callable(add_loss) else None
            if algo == "PPO":
                self.model = PPO(inputs=inputs, policy=policy, vf_coef=G.vf_coef, ent_coef=G.ent_coef)
            elif algo == "VPG":
                self.model = VPG(inputs=inputs, policy=policy, vf_coef=G.vf_coef)
            elif algo == "CPI":
                self.model = CPI(inputs=inputs, policy=policy, vf_coef=G.vf_coef, ent_coef=G.ent_coef)
            elif algo == "BC":
                self.model = BC(inputs=inputs, policy=policy)
            elif algo == "BCLearnedLoss":
                self.model = BCLearnedLoss(inputs=inputs, policy=policy, type=G.learned_loss_type)

            self.loss = self.model.loss if ext_loss is None else (self.model.loss + ext_loss)

            if not loss_only:
                if lr_rank == 0:
                    inputs.LR = lr = tf.placeholder(tf.float32, shape=[], name="LR")
                elif lr_rank == 1:
                    inputs.LR = lr = tf.placeholder(tf.float32, shape=(len(self.trainables),), name="LR")
                elif lr_rank == 2:
                    inputs.LR = lr = [tf.placeholder(tf.float32, shape=t.shape, name=f"LR_{stem(t, 2)}")
                                      for t in self.trainables]
                elif lr_rank is None:
                    lr = None
                else:
                    raise NotImplementedError(f"lr_rank = {lr_rank} is not supported. Check for programming error.")
                self.optim = Optimize(lr=lr, loss=self.loss, reports=self.model.reports,
                                      trainables=self.trainables, max_grad_norm=max_grad_norm,
                                      max_grad_clip=max_grad_clip, optimizer=optimizer)


def _mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, reduction_indices=None if axis is None else [axis], keep_dims=keepdims)


def cmaml_loss(neglogpacs, advantage):
    mean_adv = _mean(advantage)
    # we attribute adv to all workers in the style of DICE
    exploration_term = _mean(neglogpacs) * mean_adv
    return exploration_term * G.e_maml_lambda


class SingleTask:
    def __init__(self, act_space, ob_shape, trainable_map, meta_trainable_map=None, lr=None):
        # no need to go beyond despite of large G.eval_grad_steps, b/c RL samples using runner policy.

        if meta_trainable_map is None:
            meta_trainable_map = trainable_map

        self.workers = defaultlist(None)
        self.metas = defaultlist(None)

        params = defaultlist(None)
        params[0] = meta_trainable_map.copy()
        params[0].update(trainable_map)

        import gym
        assert type(act_space) is gym.spaces.Box
        act_dim, *_ = act_space.shape

        for k in range(G.n_grad_steps + 1):
            if k < G.n_grad_steps:  # 0 - 9,

                self.workers[k] = worker = make_with_custom_variables(
                    lambda: Meta(scope_name=f'inner_{k}_grad_network',
                                 act_space=act_space, ob_shape=ob_shape, algo=G.inner_alg,
                                 # do NOT pass in learning rate to inhibit the Meta.optimize operator.
                                 optimizer=G.inner_optimizer, reuse=True, trainables=list(params[k].values()),
                                 max_grad_norm=G.inner_max_grad_norm, max_grad_clip=G.inner_max_grad_clip,
                                 fix_variance=True
                                 ),  # pass in the trainable_map for proper gradient
                    params[k], f'{get_scope_name()}/inner_{k}_grad_network/'
                )

                with tf.variable_scope(f'SGD_grad_{k}'):
                    if (isinstance(lr, Sequence) and len(lr)) or (hasattr(lr, 'shape') and len(lr.shape)):
                        learn_rates = lr[k]
                    else:
                        worker.inputs.LR = lr  # this is important because this is needed by the feed_dict
                        learn_rates = [lr] * len(worker.optim.grads)
                    params[k + 1] = meta_trainable_map.copy()
                    if G.first_order:
                        params[k + 1].update({k: worker.optim.apply_grad(lr=lr, grad=tf.stop_gradient(g), var=v)
                                              for g, lr, (k, v) in
                                              zip(learn_rates, worker.optim.grads, params[k].items())})
                    else:
                        params[k + 1].update({k: worker.optim.apply_grad(lr=lr, grad=g, var=v)
                                              for g, lr, (k, v) in
                                              zip(learn_rates, worker.optim.grads, params[k].items())})

            if k == G.n_grad_steps:  # 10 or 1.
                add_loss = None if G.run_mode != 'e-maml' \
                    else lambda ADV: cmaml_loss([w.model.neglogpac for w in self.workers], ADV)
                self.meta = make_with_custom_variables(
                    lambda: Meta(scope_name="meta_network", act_space=act_space, ob_shape=ob_shape,
                                 algo=G.meta_alg, reuse=True, add_loss=add_loss, loss_only=True, )
                    , params[k], f'{get_scope_name()}/meta_network/'
                )

        # Expose as non-public API for debugging purposes
        self._params = params


def assert_match(l1, l2):
    assert len(l1) > 0
    for i, (a, b) in enumerate(zip(l1, l2)):
        assert a == b, "existing items has to be the same."
    return l1[i + 1:] if len(l1) > len(l2) else l2[i + 1:]


# Algorithm Summary
# 1. [sample] with pi(theta) `run_episode`
# 2. compute policy gradient (vanilla)
# 3. apply gradient to get \theta' using SGD
# 4. [sample] with pi(theta') `run_episode`
# 5. use PPO, compute meta gradient
# 6. sum up the PPO gradient from multiple tasks and average
# 6. apply this gradient
class E_MAML:
    gradient_sum = None
    alpha = None

    def __init__(self, ob_space, act_space):
        """
        Usage:
            self.env = env
            ob_shape = (None,) + self.env.observation_space.shape
        """
        from ml_logger import logger
        logger.upload_file(__file__)

        ob_shape = (None,) + ob_space.shape

        import gym
        assert type(act_space) is gym.spaces.Box
        act_dim, *_ = act_space.shape

        if G.meta_sgd == 'full':
            lr_rank = 2
        elif G.meta_sgd:
            lr_rank = 1
        else:
            lr_rank = 0
        # Meta holds policy, inner optimizer. Also creates an input.LR placeholder.
        self.runner = Meta(scope_name='runner', act_space=act_space, ob_shape=ob_shape, algo=G.inner_alg,
                           lr_rank=lr_rank, optimizer=G.inner_optimizer, max_grad_norm=G.inner_max_grad_norm,
                           max_grad_clip=G.inner_max_grad_clip, fix_variance=G.control_variance)

        trainables = self.runner.trainables
        runner_var_map = var_map(trainables, 'runner/')
        # note: the point of AUTO_REUSE is:
        # note:            if reuse=True, gives error when no prior is available. Otherwise always creates new.
        # note:            This yaw, only creates new when old is not available.
        self.meta_runner = Meta(scope_name="runner", act_space=act_space, ob_shape=ob_shape, algo=G.meta_alg,
                                reuse=tf.AUTO_REUSE, loss_only=True, fix_variance=G.fix_meta_variance)
        meta_trainables = self.meta_runner.trainables
        meta_runner_var_map = var_map(meta_trainables, 'runner/')
        # meta_trainables = assert_match(trainables, meta_trainables)

        self.beta = tf.placeholder(tf.float32, [], name="beta")

        print(">>>>>>>>>>> Constructing Meta Graph <<<<<<<<<<<")
        # todo: we can do multi-GPU placement of the graph here.
        self.graphs = []
        assert G.n_graphs == 1 or G.n_graphs == G.n_tasks, "graph number is 1 or equal to the number of tasks"

        if G.meta_sgd:
            assert isinstance(G.alpha, Number), "alpha need to be a scalar."
            self.alpha = []  # has to be per-layer per-block. Bias and weights require different scales.
            for k in range(G.n_grad_steps):
                with tf.variable_scope(f'learned_alpha_{k}'):
                    self.alpha.append([
                        tf.get_variable(f'alpha_{stem(t.name, 2)}', shape=t.shape if G.meta_sgd == "full" else (),
                                        initializer=tf.constant_initializer(G.alpha))
                        for t in trainables
                    ])
        else:
            self.alpha = self.runner.inputs.LR
        for t in trange(G.n_graphs):
            with tf.variable_scope(f"graph_{t}"):
                # note: should use different learning rate for each gradient step
                task_graph = SingleTask(act_space=act_space, ob_shape=ob_shape, trainable_map=runner_var_map,
                                        meta_trainable_map=meta_runner_var_map, lr=self.alpha)
                self.graphs.append(task_graph)

        all_trainables = tf.trainable_variables()  # might be controlled variables in the meta loop

        # Only do this after the meta graph has finished using policy.trainables
        # Note: stateful operators for saving to a cache and loading from it. Only used to reset runner
        # Note: Slots are not supported. Only weights.
        # fixit: all_variables might not be needed. Only that of the runner need to be cached.
        with tf.variable_scope("weight_cache"):
            self.cache = Cache(all_trainables)
            self.save_weight_cache = U.function([], [self.cache.save])
            self.load_weight_cache = U.function([], [self.cache.load])

        # Now construct the meta optimizers
        with tf.variable_scope('meta_optimizer'):
            # call gradient_sum.set_op first, then add_op. Call k times in-total.
            self.meta_grads = tf.gradients(tf.reduce_mean([task_graph.meta.loss for task_graph in self.graphs]),
                                           all_trainables)
            if G.n_graphs == 1:
                self.gradient_sum = GradientSum(all_trainables, self.meta_grads)
                grads = [c / G.n_tasks for c in self.gradient_sum.cache]
            else:
                grads = self.meta_grads

            if G.meta_max_grad_norm:  # allow 0 to be by-pass
                grads = [None if g is None else
                         g * tf.stop_gradient(G.meta_max_grad_norm / tf.maximum(G.meta_max_grad_norm, tf.norm(g)))
                         for g in grads]

            # do NOT apply gradient norm here.
            if G.meta_optimizer == "Adam":
                Optim, kwargs = tf.train.AdamOptimizer, {}
            elif G.meta_optimizer == "AdamW":
                Optim, kwargs = tf.contrib.opt.AdamWOptimizer, dict(weight_decay=0.0001)
            elif G.meta_optimizer == "SGD":
                Optim, kwargs = tf.train.GradientDescentOptimizer, {}
            else:
                raise NotImplemented(f"{G.meta_optimizer} as a meta optimizer is not implemented.")

            # Uses a different optimizer (with slots) for each step in the meta update.
            self.meta_update_ops = defaultlist(None)
            self.meta_optimizers = defaultlist(None)
            for i in range(1 if G.reuse_meta_optimizer else G.meta_n_grad_steps):
                self.meta_optimizers[i] = Optim(learning_rate=self.beta, **kwargs)
                self.meta_update_ops[i] = self.meta_optimizers[i].apply_gradients(zip(grads, all_trainables))

            self.meta_reporting_keys = self.graphs[0].meta.model.reports.keys()
            self.meta_reporting = self.graphs[0].meta.model.reports.values() if G.n_graphs == 1 else \
                [tf.reduce_mean(_) for _ in zip(*[graph.meta.model.reports.values() for graph in self.graphs])]
