import numpy as np
from gym import utils
from gym.envs import register
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding


class Controls:
    goal_velocity = 0.5
    low = 0.0
    high = 2.0

    def sample(self, goal_velocity=None):
        if goal_velocity:
            self.goal_velocity = goal_velocity
        else:
            self.goal_velocity = np.random.uniform(low=self.low, high=self.high)


class HalfCheetahGoalVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Half cheetah environment with a randomly generated goal path.
    """

    def __init__(self):
        self.controls = Controls()
        # call super init after initializing the variables.
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # xposbefore = self.model.data.qpos[0, 0]
        # change model api to work with Mujoco1.5
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        # xposafter = self.model.data.qpos[0, 0]
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 1e-1 * 0.5 * np.square(action).sum()  # add factor of 0.5, ref cbfinn.
        velocity = (xposafter - xposbefore) / self.dt
        cost_run = abs(velocity - self.controls.goal_velocity)
        reward = reward_ctrl - cost_run
        done = False
        return ob, reward, done, dict(cost_run=cost_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            # self.model.data.qpos.flat[1:],
            self.data.qpos[1:],
            # self.model.data.qvel.flat,
            self.data.qvel,
        ])

    def set_goal_velocity(self, goal_velocity=None):
        # print('***** goal velocity **********>>', goal_velocity)
        self.controls.sample(goal_velocity=goal_velocity)

    def get_goal_velocity(self):  # only for debugging
        return self.controls.goal_velocity

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


register(
    id='HalfCheetahGoalVel-v0',
    # todo: use module.sub_module:ClassName syntax to work with rcall and cloudpickle.
    # entry_point=lambda: HalfCheetahGoalVelEnv(),
    entry_point="e_maml_tf.custom_vendor.half_cheetah_goal_velocity:HalfCheetahGoalVelEnv",
    kwargs={},
    max_episode_steps=200,
    reward_threshold=4800.0,
)
