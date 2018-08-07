import multiprocessing
from params_proto import cli_parse, Proto

ALLOWED_ALGS = "rl_algs.PPO", "rl_algs.VPG", "PPO", "VPG"
DIR_TEMPLATE = "{now:%Y-%m-%d}/e_maml_tf/" \
               "{G.run_mode}-{G.env_name}-n_grad({G.n_grad_steps})" \
               "-{G.inner_alg}-{G.inner_optimizer}" \
               "-{G.meta_alg}-{G.meta_optimizer}-alpha({G.alpha})-beta({G.beta})" \
               "-n_graphs({G.n_graphs})-env_norm({G.normalize_env})" \
               "-grad_norm({G.inner_max_grad_norm})-meta_grad_norm({G.meta_max_grad_norm})-{now:%H%M%S}-{now:%f}"

from datetime import datetime

now = datetime.now()


@cli_parse
class RUN:
    log_dir = ""
    log_prefix = 'e-maml-debug'


def config_run(**_G):
    G.update(_G)
    from datetime import datetime
    now = datetime.now()
    RUN.log_prefix = DIR_TEMPLATE.format(now=now, G=G)


# decorator help generate a as command line parser.
@cli_parse
class G:
    # Termination conditions
    term_loss_threshold = 100
    term_reward_threshold = -8000.0

    run_mode = "maml"  # type:  "Choose between maml and e_maml. Switches the loss function used for training"
    e_maml_lambda = Proto(1.0, help="The scaling factor for the E-MAML term")
    # env_name = 'HalfCheetah-v2'  # type:  "Name of the task environment"
    env_name = 'HalfCheetahGoalDir-v0'  # type:  "Name of the task environment"
    start_seed = Proto(0, help="seed for initialization of each game")
    render = False
    n_cpu = multiprocessing.cpu_count() * 2  # type: "number of threads used"

    # (E_)MAML Training Parameters
    n_tasks = Proto(20, help="40 for locomotion, 20 for 2D navigation ref:cbfinn")
    n_graphs = Proto(1, help="number of parallel graphs for multi-device parallelism. Hard coded to 1 atm.")
    n_grad_steps = 5  # type:  "number of gradient descent steps for the worker." #TODO change back to 1
    meta_n_grad_steps = Proto(1, help="number of gradient descent steps for the meta algorithm.")
    reuse_meta_optimizer = Proto(True, help="Whether to use the same AdamW optimizer for all "
                                            "meta gradient steps. MUCH FASTER to initialize with [True].")
    eval_grad_steps = Proto(list(range(n_grad_steps + 1)),
                            help="the gradient steps at which we evaluate the policy. Used to make pretty plots.")

    bias_dim = Proto(20, help="the input bias variable dimension that breaks the input symmetry")
    # 40k per task (action, state) tuples, or 20k (per task) if you have 10/20 meta tasks
    n_parallel_envs = 40  # type:  "Number of parallel envs in minibatch. The SubprocVecEnv batch_size."
    batch_timesteps = 100  # type:  "max_steps for each episode, used to set env._max_steps parameter"

    epoch_init = Proto(0, help="the epoch to start with.")
    n_epochs = 800  # type:  "Number of epochs"
    eval_interval = Proto(None, help="epoch interval for evaluation.")
    eval_num_envs = Proto(n_parallel_envs, help="default to same as sampling envs")
    eval_timesteps = Proto(50, help="batch size for the evaluation RL runs")

    record_movie_interval = 500
    start_movie_after_epoch = 700
    render_num_envs = Proto(10, help="keep small b/c rendering is slow")
    movie_timesteps = 100  # type: "now runs in batch mode"
    start_checkpoint_after_epoch = Proto(200, help="epoch at which start saving checkpoints.")
    checkpoint_interval = Proto(None, help="the frequency for saving checkpoints on the policy")
    load_from_checkpoint = Proto(None, help="the path to the checkpoint file (saved by logger) to be loaded at the"
                                            " beginning of the training session. Also includes the learned loss, "
                                            "and learned learning rates if available.")

    # RL sampling settings
    reset_on_start = Proto(False, help="reset the environment at the beginning of each episode. "
                                       "Do NOT use this when using SubProcessVecEnv")

    # behavior cloning
    mask_meta_bc_data = Proto(False, help='masking the state space for one-shot imitation baseline')
    # bc_eval_timesteps = Proto(100, help="number of timesteps for evaluation")
    episode_subsample = Proto(1, help='the subsampling ratio for episodic training dataset. Active under episode mode')
    sample_limit = Proto(None, help='the number of timesteps uses in behavior cloning algorithm.')
    k_fold = Proto(5, help='the k-fold cross validation')

    env_max_timesteps = Proto(0, help="max_steps for each episode, used to set env._max_steps parameter. 0 to use "
                                      "gym default.")
    single_sampling = 0  # type:  "flag for running a single sampling step. 1 ON, 0 OFF"
    baseline = Proto('linear', help="using the critic as the baseline")
    use_gae = Proto(True, help="flag to turn GAE on and off")
    # GAE runner options
    gamma = Proto(0.995, help="GAE gamma")
    lam = Proto(0.97, help="GAE lambda")
    # Imperfect Demonstration Options
    # imperfect_demo = Proto(None, help='flag to turn on the systematic noise for the imperfect demonstration')
    # demo_offset_abs = Proto(None, help='size of the systematic offset to the goal position in expert demo')
    # demo_noise_scale = Proto(None, help='scale of the noise added to the goal position in expert demo')

    # MAML Options
    first_order = Proto(False, help="Whether to stop gradient calculation during meta-gradient calculation")
    alpha = 0.05  # type:  "worker learning rate. use 0.1 for first step, 0.05 afterward ref:cbfinn"
    meta_sgd = Proto(None, help='One of [None, True, "full"]. When full learns alpha same shape as tensors.')
    beta = 0.01  # type:  "meta learning rate"
    inner_alg = "VPG"  # type:  '"PPO" or "VPG", "rl_algs.VPG" or "rl_algs.PPO" for rl_algs baselines'
    learned_loss_type = None
    inner_optimizer = "SGD"  # type:  '"AdamW", "Adam", or "SGD"'
    meta_alg = "PPO"  # type:  "PPO or TRPO, TRPO is not yet implemented."
    meta_optimizer = "AdamW"  # type:  '"AdamW", "Adam" or "SGD"'
    activation = "tanh"
    n_layers = 4  # type: "the number of hidden layers for the policy network. Sometimes, bigger, is better"
    hidden_size = 64  # type: "hidden size for the MLP policy"

    # Model options
    use_k_index = Proto(False, help="whether to wrap k_index around the environment. Helps for the value baseline")
    normalize_env = False  # type: "normalize the environment"
    vf_coef = 0.5  # type:  "loss weighing coefficient for the value function loss. with the VPG loss being 1.0"
    ent_coef = 0.01  # type:  "PPO entropy coefficient"
    inner_max_grad_norm = 1.0  # type:  "PPO maximum gradient norm"
    meta_max_grad_norm = 1.0  # type:  "PPO maximum gradient norm"
    inner_max_grad_clip = Proto(None, help="maximum gradient clip")
    meta_max_grad_clip = Proto(None, help="maximum gradient clip")
    clip_range = Proto(0.2, help="PPO clip_range parameter")

    # policy parameters
    init_logstd = Proto(0, help="initial log standard deviation of the gaussian policy")
    control_variance = Proto(False, help='flag for fixing the variance of the policy for the inner worker. Helps '
                                         'prevent inner adaptation from gaining too much from reducing variance.')
    fix_meta_variance = Proto(False, help="flag for fixing the meta runner's variance.")
    std_l2_coef = Proto(0, help="the regularization coefficient for the standard deviation")

    # Grid World config parameters
    change_colors = 0  # type:  "shuffle colors of the board game"
    change_dynamics = 0  # type:  'shuffle control actions (up down, left right) of the game'


@cli_parse
class Reporting:
    report_mean = False  # type:  "plot the mean instead of the total reward per episode"
    log_device_placement = False


@cli_parse
class DEBUG:
    """To debug:
    Set debug_params = 1,
    set debug_apply_gradient = 1.
    Then the gradient ratios between the worker and the meta runner should be print out, and they should be 1.
    Otherwise, the runner model is diverging from the meta network.
    """
    no_weight_reset = Proto(0, help="flag to turn off the caching and resetting the weights")
    no_task_resample = Proto(0, help="by-pass task re-sample")
