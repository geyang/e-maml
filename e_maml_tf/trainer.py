import tensorflow as tf
from collections import defaultdict
from ml_logger import logger, metrify
from e_maml_tf.meta_rl_tasks import MetaRLTasks
from termcolor import colored

from e_maml_tf.ge_utils import stem
from e_maml_tf.packages.schedules import Schedule
from e_maml_tf.sampler import path_gen_fn, paths_process
from .e_maml_ge import E_MAML
from .config import G, DEBUG, Reporting
import numpy as np
from e_maml_tf.algos import vpg, ppo2, cpi, bc, bc_learned_loss


def train_supervised_maml(*, k_tasks=1, maml: E_MAML):
    # env used for evaluation purposes only.
    if G.meta_sgd:
        assert maml.alpha is not None, "Coding Mistake if meta_sgd is trueful but maml.alpha is None."

    assert G.n_tasks >= k_tasks, f"Is this intended? You probably want to have " \
                                 f"meta-batch({G.n_tasks}) >= k_tasks({k_tasks})."

    sess = tf.get_default_session()

    epoch_ind, pref = -1, ""
    while epoch_ind < G.n_epochs:
        # for epoch_ind in range(G.n_epochs + 1):
        logger.flush()
        logger.split()

        is_bc_test = (pref != "test/" and G.eval_interval and epoch_ind % G.eval_interval == 0)
        pref = "test/" if is_bc_test else ""
        epoch_ind += 0 if is_bc_test else 1

        if G.meta_sgd:
            alpha_lr = sess.run(maml.alpha)  # only used in the runner.
            logger.log(metrics={f"alpha_{i}/{stem(t.name, 2)}": a
                                for i, a_ in enumerate(alpha_lr)
                                for t, a in zip(maml.runner.trainables, a_)}, silent=True)
        else:
            alpha_lr = G.alpha.send(epoch_ind) if isinstance(G.alpha, Schedule) else np.array(G.alpha)
            logger.log(alpha=metrify(alpha_lr), epoch=epoch_ind, silent=True)

        beta_lr = G.beta.send(epoch_ind) if isinstance(G.beta, Schedule) else np.array(G.beta)
        logger.log(beta=metrify(beta_lr), epoch=epoch_ind, silent=True)

        if G.checkpoint_interval and epoch_ind % G.checkpoint_interval == 0:
            yield "pre-update-checkpoint", epoch_ind

        # Compute updates for each task in the batch
        # 0. save value of variables
        # 1. sample
        # 2. gradient descent
        # 3. repeat step 1., 2. until all gradient steps are exhausted.
        batch_data = defaultdict(list)

        maml.save_weight_cache()
        load_ops = [] if DEBUG.no_weight_reset else [maml.cache.load]

        feed_dict = {}
        for task_ind in range(k_tasks if is_bc_test else G.n_tasks):
            graph_branch = maml.graphs[0] if G.n_graphs == 1 else maml.graphs[task_ind]
            if G.n_graphs == 1:
                gradient_sum_op = maml.gradient_sum.set_op if task_ind == 0 else maml.gradient_sum.add_op

            """
            In BC mode, we don't have an environment. The sampling is handled here then fed to the sampler.
            > task_spec = dict(index=0)
            
            Here we make the testing more efficient.
            """
            if not DEBUG.no_task_resample:
                if not is_bc_test:
                    task_spec = dict(index=np.random.randint(0, k_tasks))
                elif task_ind < k_tasks:
                    task_spec = dict(index=task_ind % k_tasks)
                else:
                    raise RuntimeError('should never hit here.')

            for k in range(G.n_grad_steps + 1):  # 0 - 10 <== last one being the maml policy.

                # for imitation inner loss, we still sample trajectory for evaluation purposes, but
                # replace it with the demonstration data for learning
                if k < G.n_grad_steps:
                    p = p if G.single_sampling and k > 0 else \
                        bc.sample_demonstration_data(task_spec, key=("eval" if is_bc_test else None))
                elif k == G.n_grad_steps:
                    # note: use meta bc samples.
                    p = bc.sample_demonstration_data(task_spec, key="meta")
                else:
                    raise Exception('Implementation error. Should never reach this line.')

                _p = {k: v for k, v in p.items() if k != "ep_info"}

                if k < G.n_grad_steps:
                    # note: under meta-SGD mode, the runner needs the k^th learning rate.
                    _lr = alpha_lr[k] if G.meta_sgd else alpha_lr

                    runner_feed_dict = \
                        path_to_feed_dict(inputs=maml.runner.inputs, paths=_p, lr=_lr)
                    # todo: optimize `maml.meta_runner` if k >= G.n_grad_steps.
                    loss, *_, __ = maml.runner.optim.run_optimize(feed_dict=runner_feed_dict)
                    runner_feed_dict.clear()

                    for key, value in zip(maml.runner.model.reports.keys(), [loss, *_]):
                        batch_data[pref + f"grad_{k}_step_{key}"].append(value)
                        logger.log_key_value(pref + f"task_{task_ind}_grad_{k}_{key}", value, silent=True)

                    if loss > G.term_loss_threshold:  # todo: make this batch-based instead of on single episode
                        err = pref + "episode loss blew up:", loss, "terminating training."
                        logger.log_line(colored(err, "red"), flush=True)
                        raise RuntimeError('loss is TOO HIGH. Terminating the experiment.')

                    # fixit: has bug when using fixed learning rate. Still needs to get learning rate from placeholder
                    feed_dict.update(path_to_feed_dict(inputs=graph_branch.workers[k].inputs, paths=_p))
                elif k == G.n_grad_steps:
                    yield_keys = dict(
                        movie=G.record_movie_interval and epoch_ind >= G.start_movie_after_epoch and
                              epoch_ind % G.record_movie_interval == 0,
                        eval=is_bc_test
                    )
                    if np.fromiter(yield_keys.values(), bool).any():
                        yield yield_keys, epoch_ind, task_spec
                    if is_bc_test:
                        if load_ops:
                            tf.get_default_session().run(load_ops)
                        continue  # do NOT meta learn from test samples.

                    # we don't treat the meta_input the same way even though we could. This is more clear to read.
                    # note: feed in the learning rate only later.
                    feed_dict.update(path_to_feed_dict(inputs=graph_branch.meta.inputs, paths=_p))

                    if G.n_graphs == 1:
                        # load from checkpoint before computing the meta gradient\nrun gradient sum operation
                        if load_ops:
                            tf.get_default_session().run(load_ops)
                        # note: meta reporting should be run here. Not supported for simplicity. (need to reduce across
                        # note: tasks, and can not be done outside individual task graphs.
                        if G.meta_sgd is None:
                            feed_dict[maml.alpha] = alpha_lr
                        tf.get_default_session().run(gradient_sum_op, feed_dict)
                        feed_dict.clear()

                    if load_ops:
                        tf.get_default_session().run(load_ops)

        if is_bc_test:
            continue  # do NOT meta learn from test samples.

        if G.meta_sgd is None:
            feed_dict[maml.alpha] = alpha_lr

        if G.n_graphs == 1:
            assert G.meta_n_grad_steps == 1, "ERROR: Can only run 1 meta gradient step with a single graph."
            # note: remove meta reporting b/c meta report should be in each task in this case.
            tf.get_default_session().run(maml.meta_update_ops[0], {maml.beta: beta_lr})
        else:
            assert feed_dict, "ERROR: It is likely that you jumped here from L:178."
            feed_dict[maml.beta] = beta_lr
            for i in range(G.meta_n_grad_steps):
                update_op = maml.meta_update_ops[0 if G.reuse_meta_optimizer else i]
                *reports, _ = tf.get_default_session().run(maml.meta_reporting + [update_op], feed_dict)
                if i not in (0, G.meta_n_grad_steps - 1):
                    continue
                for key, v in zip(maml.meta_reporting_keys, reports):
                    logger.log_key_value(pref + f"grad_{G.n_grad_steps + i}_step_{key}", v, silent=True)

            feed_dict.clear()

        tf.get_default_session().run(maml.cache.save)

        # Now compute the meta gradients.
        # note: runner shares variables with the MAML graph. Reload from state_dict
        # note: if max_grad_step is the same as n_grad_steps then no need here.

        dt = logger.split()
        logger.log_line('Timer Starts...' if dt is None else f'{dt:0.2f} sec/epoch')
        logger.log(dt_epoch=dt or np.nan, epoch=epoch_ind)

        for key, arr in batch_data.items():
            reduced = np.array(arr).mean()
            logger.log_key_value(key, reduced)


def train_maml(*, n_tasks: int, tasks: MetaRLTasks, maml: E_MAML):
    if not G.inner_alg.startswith("BC"):
        path_gen = path_gen_fn(env=tasks.envs, policy=maml.runner.policy, start_reset=G.reset_on_start)
        next(path_gen)

    meta_path_gen = path_gen_fn(env=tasks.envs, policy=maml.meta_runner.policy, start_reset=G.reset_on_start)
    next(meta_path_gen)

    if G.load_from_checkpoint:
        # todo: add variable to checkpoint
        # todo: set the epoch_ind starting point here.
        logger.load_variables(G.load_from_checkpoint)

    if G.meta_sgd:
        assert maml.alpha is not None, "Coding Mistake if meta_sgd is trueful but maml.alpha is None."

    max_episode_length = tasks.spec.max_episode_steps

    sess = tf.get_default_session()
    epoch_ind, prefix = G.epoch_init - 1, ""
    while epoch_ind < G.epoch_init + G.n_epochs:
        logger.flush()
        logger.split()

        is_bc_test = (prefix != "test/" and G.eval_interval and epoch_ind % G.eval_interval == 0)
        prefix = "test/" if is_bc_test else ""
        epoch_ind += 0 if is_bc_test else 1

        if G.meta_sgd:
            alpha_lr = sess.run(maml.alpha)  # only used in the runner.
            logger.log(metrics={f"alpha_{i}/{stem(t.name, 2)}": a
                                for i, a_ in enumerate(alpha_lr)
                                for t, a in zip(maml.runner.trainables, a_)}, silent=True)
        else:
            alpha_lr = G.alpha.send(epoch_ind) if isinstance(G.alpha, Schedule) else np.array(G.alpha)
            logger.log(alpha=metrify(alpha_lr), epoch=epoch_ind, silent=True)

        beta_lr = G.beta.send(epoch_ind) if isinstance(G.beta, Schedule) else np.array(G.beta)
        clip_range = G.clip_range.send(epoch_ind) if isinstance(G.clip_range, Schedule) else np.array(G.clip_range)
        logger.log(beta=metrify(beta_lr), clip_range=metrify(clip_range), epoch=epoch_ind, silent=True)

        batch_timesteps = G.batch_timesteps.send(epoch_ind) \
            if isinstance(G.batch_timesteps, Schedule) else G.batch_timesteps

        # Compute updates for each task in the batch
        # 0. save value of variables
        # 1. sample
        # 2. gradient descent
        # 3. repeat step 1., 2. until all gradient steps are exhausted.
        batch_data = defaultdict(list)

        maml.save_weight_cache()
        load_ops = [] if DEBUG.no_weight_reset else [maml.cache.load]

        if G.checkpoint_interval and epoch_ind % G.checkpoint_interval == 0 \
                and not is_bc_test and epoch_ind >= G.start_checkpoint_after_epoch:
            cp_path = f"checkpoints/variables_{epoch_ind:04d}.pkl"
            logger.log_line(f'saving checkpoint {cp_path}')
            # note: of course I don't know that are all of the trainables at the moment.
            logger.save_variables(tf.trainable_variables(), path=cp_path)

        feed_dict = {}
        for task_ind in range(n_tasks if is_bc_test else G.n_tasks):
            graph_branch = maml.graphs[0] if G.n_graphs == 1 else maml.graphs[task_ind]
            if G.n_graphs == 1:
                gradient_sum_op = maml.gradient_sum.set_op if task_ind == 0 else maml.gradient_sum.add_op

            print(f"task_ind {task_ind}...")
            if not DEBUG.no_task_resample:
                if not is_bc_test:
                    print(f'L250: sampling task')
                    tasks.sample()
                elif task_ind < n_tasks:
                    task_spec = dict(index=task_ind % n_tasks)
                    print(f'L254: sampling task {task_spec}')
                    tasks.sample(**task_spec)
                else:
                    raise RuntimeError('should never hit here.')

            for k in range(G.n_grad_steps + 1):  # 0 - 10 <== last one being the maml policy.
                _is_new = False
                # for imitation inner loss, we still sample trajectory for evaluation purposes, but
                # replace it with the demonstration data for learning
                if k < G.n_grad_steps:
                    if G.inner_alg.startswith("BC"):
                        p = p if G.single_sampling and k > 0 else \
                            bc.sample_demonstration_data(tasks.task_spec, key=("eval" if is_bc_test else None))
                    else:
                        p, _is_new = path_gen.send(batch_timesteps), True
                elif k == G.n_grad_steps:
                    if G.meta_alg.startswith("BC"):
                        # note: use meta bc samples.
                        p = bc.sample_demonstration_data(tasks.task_spec, key="meta")
                    else:
                        p, _is_new = meta_path_gen.send(batch_timesteps), True
                else:
                    raise Exception('Implementation error. Should never reach this line.')

                if k in G.eval_grad_steps:
                    _ = path_gen if k < G.n_grad_steps else meta_path_gen
                    p_eval = p if _is_new else _.send(G.eval_timesteps)
                    # reporting on new trajectory samples
                    avg_r = p_eval['ep_info']['reward'] if G.normalize_env else np.mean(p_eval['rewards'])
                    episode_r = avg_r * max_episode_length  # default horizon for HalfCheetah

                    if episode_r < G.term_reward_threshold:  # todo: make this batch-based instead of on single episode
                        logger.log_line("episode reward is too low: ", episode_r, "terminating training.", flush=True)
                        raise RuntimeError('AVERAGE REWARD TOO LOW. Terminating the experiment.')

                    batch_data[prefix + f"grad_{k}_step_reward"].append(avg_r if Reporting.report_mean else episode_r)
                    if k in G.eval_grad_steps:
                        logger.log_key_value(prefix + f"task_{task_ind}_grad_{k}_reward", episode_r, silent=True)

                _p = {k: v for k, v in p.items() if k != "ep_info"}

                if k < G.n_grad_steps:
                    # note: under meta-SGD mode, the runner needs the k^th learning rate.
                    _lr = alpha_lr[k] if G.meta_sgd else alpha_lr

                    # clip_range is not used in BC mode. but still passed in.
                    runner_feed_dict = \
                        path_to_feed_dict(inputs=maml.runner.inputs, paths=_p, lr=_lr,
                                          baseline=G.baseline, gamma=G.gamma, use_gae=G.use_gae, lam=G.lam,
                                          horizon=max_episode_length, clip_range=clip_range)
                    # todo: optimize `maml.meta_runner` if k >= G.n_grad_steps.
                    loss, *_, __ = maml.runner.optim.run_optimize(feed_dict=runner_feed_dict)
                    runner_feed_dict.clear()

                    for key, value in zip(maml.runner.model.reports.keys(), [loss, *_]):
                        batch_data[prefix + f"grad_{k}_step_{key}"].append(value)
                        logger.log_key_value(prefix + f"task_{task_ind}_grad_{k}_{key}", value, silent=True)

                    if loss > G.term_loss_threshold:  # todo: make this batch-based instead of on single episode
                        logger.log_line(prefix + "episode loss blew up:", loss, "terminating training.", flush=True)
                        raise RuntimeError('loss is TOO HIGH. Terminating the experiment.')

                    # done: has bug when using fixed learning rate. Needs the learning rate as input.
                    feed_dict.update(  # do NOT pass in the learning rate because the graph already includes those.
                        path_to_feed_dict(inputs=graph_branch.workers[k].inputs, paths=_p,
                                          lr=None if G.meta_sgd else alpha_lr,  # but do with fixed alpha
                                          horizon=max_episode_length,
                                          baseline=G.baseline, gamma=G.gamma, use_gae=G.use_gae, lam=G.lam,
                                          clip_range=clip_range))

                elif k == G.n_grad_steps:
                    yield_keys = dict(
                        movie=epoch_ind >= G.start_movie_after_epoch and epoch_ind % G.record_movie_interval == 0,
                        eval=is_bc_test
                    )
                    if np.fromiter(yield_keys.values(), bool).any():
                        yield yield_keys, epoch_ind, tasks.task_spec
                    if is_bc_test:
                        if load_ops:  # we need to reset the weights. Otherwise the world would be on fire.
                            tf.get_default_session().run(load_ops)
                        continue  # do NOT meta learn from test samples.

                    # we don't treat the meta_input the same way even though we could. This is more clear to read.
                    # note: feed in the learning rate only later.
                    feed_dict.update(  # do NOT need learning rate
                        path_to_feed_dict(inputs=graph_branch.meta.inputs, paths=_p,
                                          horizon=max_episode_length,
                                          baseline=G.baseline, gamma=G.gamma, use_gae=G.use_gae, lam=G.lam,
                                          clip_range=clip_range))

                    if G.n_graphs == 1:
                        # load from checkpoint before computing the meta gradient\nrun gradient sum operation
                        if load_ops:
                            tf.get_default_session().run(load_ops)
                        # note: meta reporting should be run here. Not supported for simplicity. (need to reduce across
                        # note: tasks, and can not be done outside individual task graphs.
                        if G.meta_sgd is None:  # note: copied from train_supervised_maml, not tested
                            feed_dict[maml.alpha] = alpha_lr
                        tf.get_default_session().run(gradient_sum_op, feed_dict)
                        feed_dict.clear()

                    if load_ops:
                        tf.get_default_session().run(load_ops)

        if is_bc_test:
            continue  # do NOT meta learn from test samples.

        # note: copied from train_supervised_maml, not tested
        if G.meta_sgd is None:
            feed_dict[maml.alpha] = alpha_lr

        if G.n_graphs == 1:
            assert G.meta_n_grad_steps == 1, "ERROR: Can only run 1 meta gradient step with a single graph."
            # note: remove meta reporting b/c meta report should be in each task in this case.
            tf.get_default_session().run(maml.meta_update_ops[0], {maml.beta: beta_lr})
        else:
            assert feed_dict, "ERROR: It is likely that you jumped here from L:178."
            feed_dict[maml.beta] = beta_lr
            for i in range(G.meta_n_grad_steps):
                update_op = maml.meta_update_ops[0 if G.reuse_meta_optimizer else i]
                *reports, _ = tf.get_default_session().run(maml.meta_reporting + [update_op], feed_dict)
                if i not in (0, G.meta_n_grad_steps - 1):
                    continue
                for key, v in zip(maml.meta_reporting_keys, reports):
                    logger.log_key_value(prefix + f"grad_{G.n_grad_steps + i}_step_{key}", v, silent=True)

            feed_dict.clear()

        tf.get_default_session().run(maml.cache.save)

        # Now compute the meta gradients.
        # note: runner shares variables with the MAML graph. Reload from state_dict
        # note: if max_grad_step is the same as n_grad_steps then no need here.

        dt = logger.split()
        logger.log_line('Timer Starts...' if dt is None else f'{dt:0.2f} sec/epoch')
        logger.log(dt_epoch=dt or np.nan, epoch=epoch_ind)

        for key, arr in batch_data.items():
            reduced = np.array(arr).mean()
            logger.log_key_value(key, reduced)

        logger.flush()


def path_to_feed_dict(*, inputs, paths, lr=None, **rest):
    from e_maml_tf.sampler import paths_process
    if isinstance(inputs, vpg.Inputs):
        paths = paths_process(paths, **rest)
        return vpg.path_to_feed_dict(inputs=inputs, paths=paths, lr=lr)  # kl limit etc
    elif isinstance(inputs, ppo2.Inputs):
        paths = paths_process(paths, **rest)
        return ppo2.path_to_feed_dict(inputs=inputs, paths=paths, lr=lr, **rest)  # kl limit etc
    elif isinstance(inputs, cpi.Inputs):
        paths = paths_process(paths, **rest)
        return cpi.path_to_feed_dict(inputs=inputs, paths=paths, lr=lr, **rest)  # kl limit etc
    elif isinstance(inputs, bc.Inputs):
        return bc.path_to_feed_dict(inputs=inputs, paths=paths, lr=lr, **rest)  # kl limit etc
    elif isinstance(inputs, bc_learned_loss.Inputs):
        return bc_learned_loss.path_to_feed_dict(inputs=inputs, paths=paths, lr=lr, **rest)  # kl limit etc
    else:
        raise NotImplementedError("Input type is not recognised")


# debug only
def eval_tensors(*, variable, feed_dict):
    return tf.get_default_session().run(variable, feed_dict)
