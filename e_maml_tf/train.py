import tensorflow as tf
from ml_logger import logger

from e_maml_tf.meta_rl_tasks import MetaRLTasks
from e_maml_tf import config
from e_maml_tf.e_maml_ge import E_MAML
from e_maml_tf.trainer import train_maml


def run_e_maml(_G=None, _DEBUG=None):
    import baselines.common.tf_util as U
    if _G is not None:
        config.G.update(_G)
    if _DEBUG is not None:
        config.DEBUG.update(_DEBUG)

    # todo: let's take the control of the log director away from the train script. It should all be set from outside.
    # done: now this is set in the runner thunk.
    # logger.configure(log_directory=config.RUN.log_dir, prefix=config.RUN.log_prefix)
    logger.log_params(
        G=vars(config.G),
        Reporting=vars(config.Reporting),
        DEBUG=vars(config.DEBUG)
    )
    logger.upload_file(__file__)

    tasks = MetaRLTasks(env_name=config.G.env_name, batch_size=config.G.n_parallel_envs,
                        start_seed=config.G.start_seed,
                        log_directory=(config.RUN.log_directory + "/{seed}") if config.G.render else None,
                        max_steps=config.G.env_max_timesteps)

    # sess_config = tf.ConfigProto(log_device_placement=config.Reporting.log_device_placement)
    # with tf.Session(config=sess_config), tf.device('/gpu:0'), tasks:
    graph = tf.Graph()
    with graph.as_default(), U.make_session(num_cpu=config.G.n_cpu), tasks:
        maml = E_MAML(ob_space=tasks.envs.observation_space, act_space=tasks.envs.action_space)

        U.initialize()

        import gym
        from rl.helpers import unbatch_policy, render_gen_fn

        eval_env = gym.make(config.G.env_name)

        if config.G.use_k_index:
            from e_maml_tf.wrappers.k_index import k_index
            eval_env = k_index(eval_env)

        _policy = unbatch_policy(maml.runner.policy)

        # todo: use batch-mode to accelerate rendering.
        rend_gen = render_gen_fn(_policy, eval_env, stochastic=False, width=640, height=480, reset_on_done=True)

        _ep_ind, _hook_cache = None, {}
        train_iter = train_maml(n_tasks=config.G.n_tasks, tasks=tasks, maml=maml)
        while True:
            try:
                status, epoch, task_spec, *_ = next(train_iter)

                t_id = task_spec['index']
                if epoch != _ep_ind:
                    _hook_cache.clear()
                _ep_ind = epoch

                if status.startswith('grad-') and status.endswith('movie'):
                    k, = _
                    hook = f"{config.G.env_name}_{epoch:04d}_k({k})_t({t_id})"
                    if hook in _hook_cache:
                        continue
                    _hook_cache[hook] = True

                    eval_env.sample_task(**task_spec)
                    movie = [next(rend_gen) for _ in range(config.G.movie_timesteps)]
                    logger.log_video(movie, "videos/" + hook + ".mp4", fps=30)
                    del movie
                    # samples = [next(sample_gen) for _ in range(config.G.movie_timesteps)]
                    # logger.log_data(samples, hook + ".pkl")

                hook = f"{config.G.env_name}_{epoch:04d}_t({t_id})"
                if status == 'post-update-movie' and hook not in _hook_cache:
                    _hook_cache[hook] = True
                    eval_env.sample_task(**task_spec)
                    movie = [next(rend_gen) for _ in range(config.G.movie_timesteps)]
                    logger.log_video(movie, "videos/" + hook + ".mp4", fps=30)
                    del movie
                    # samples = [next(sample_gen) for _ in range(config.G.movie_timesteps)]
                    # logger.log_data(samples, hook + ".pkl")

            except StopIteration:
                break
        logger.flush()

    tf.reset_default_graph()


def launch(**_G):
    import traceback
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)

    try:
        config.config_run(**_G)
        run_e_maml(_G)
    except Exception as e:
        tb = traceback.format_exc()
        logger.log_line(tb)
        raise e


if __name__ == '__main__':
    config.RUN.log_prefix = "alpha-0-check"
    launch()
