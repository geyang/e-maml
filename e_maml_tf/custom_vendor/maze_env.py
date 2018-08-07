import copy
import os
from glob import glob

import numpy as np
from gym import spaces, Env, utils
from gym.envs import register


class MazeEnv(Env, utils.EzPickle):
    _action_set = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], np.int32)

    def _seed(self, seed):
        self.task_rng = np.random.RandomState(seed)

    def __init__(self, batch_size, path, n_episodes=5, episode_horizon=12, seed=69, num_envs=None):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-1, 1, 15)  # new(1), rew(1), onehot act(4), obs(9)
        self._mazes = np.load(path)['arr_0']
        self.num_envs = num_envs
        self.reward_range = [-100, 100]
        self.metadata = dict()

        # maze id (1), ep count (1), time count (1), cur loc (2), goal(2), start(2)
        self._state = np.zeros([batch_size, 9], np.int32)
        self._alive_cost = 1.0 / episode_horizon
        self.batch_size, self.n_episodes, self.episode_horizon = batch_size, n_episodes, episode_horizon
        self._state_for_reset = copy.deepcopy(self._state)
        # self.init_pos_rng = random.Random(init_pos_seed)
        self._seed(seed)
        self.reset_task()

        utils.EzPickle.__init__(self)

    def _close(self):
        pass

    def _reset(self, dones=None):
        if dones is None:
            dones = np.ones(self.batch_size, np.bool)

        batch_size = np.sum(dones)
        self._state = copy.deepcopy(self._state_for_reset)
        obs = np.zeros((batch_size,) + self.observation_space.shape, np.float32)
        obs[:, 0] = 1
        obs[:, 6:] = self._get_obs()[dones]
        return obs.squeeze()

    def reset_task(self, dones=None):
        if dones is None:
            dones = np.ones(self.batch_size, np.bool)

        batch_size = np.sum(dones)
        maze_idx = self.task_rng.randint(len(self._mazes), size=batch_size)
        # maze_idx = self.mazes_fixed
        starts, goals = [], []
        for i in maze_idx:
            locs = list(zip(*np.where(~self._mazes[i])))
            # starts.append(self.starts_fixed)
            # goals.append(self.goals_fixed)
            starts.append(locs[self.task_rng.randint(len(locs))])
            goals.append(locs[self.task_rng.randint(len(locs))])

        self._state[dones, 0] = maze_idx
        self._state[dones, 1:3] = 0
        self._state[dones, 3:5] = np.array(starts)
        self._state[dones, 5:7] = np.array(goals)
        self._state[dones, 7:9] = np.array(starts)
        self._state_for_reset = copy.deepcopy(self._state)

        obs = np.zeros((batch_size,) + self.observation_space.shape, np.float32)
        obs[:, 0] = 1
        obs[:, 6:] = self._get_obs()[dones]
        return obs.squeeze()

    def _step(self, actions):
        t = self._state[:, 2]
        next_loc = self._state[:, 3:5] + self._action_set[actions]
        hit_wall = self._mazes[self._state[:, 0], next_loc[:, 0], next_loc[:, 1]]

        self._state[~hit_wall, 3:5] = next_loc[~hit_wall]
        t[:] += 1

        at_goal = np.equal(self._state[:, 3:5], self._state[:, 5:7]).all(1)
        finished_episode = np.equal(t, self.episode_horizon) | at_goal
        t[finished_episode] = 0
        self._state[finished_episode, 1] += 1
        self._state[finished_episode, 3:5] = self._state[finished_episode, 7:9]

        rewards = (1 + self._alive_cost) * at_goal - 1e-3 * hit_wall - self._alive_cost
        dones = np.equal(self._state[:, 1], self.n_episodes)

        obs = np.zeros((self.batch_size,) + self.observation_space.shape, np.float32)
        obs[:, 0] = finished_episode
        obs[:, 1] = rewards
        obs[np.arange(1), 2 + actions] = 1.0
        obs[:, 6:] = self._get_obs()
        return obs.squeeze(), rewards.squeeze(), dones.squeeze(), dict()

    def _get_obs(self):
        x, y = self._state[:, 3:5].T
        dx, dy = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing='ij')
        xi, yi = x[:, None, None] + dx, y[:, None, None] + dy
        mi = self._state[:, :1, None]
        obs = self._mazes[mi, xi, yi].reshape((-1, 9))
        if obs[:, 4].any():
            raise ValueError
        return obs


MAZE_DATA_PATH = glob('../../krazy_grid_world/maze_data/*.npz')  # use as assertion base
directory = os.path.dirname(__file__)
MAZES = {
    "Maze10-v0": os.path.join(directory, '../../krazy_grid_world/maze_data/mazes_10k_10x10.npz'),
    "Maze20-v0": os.path.join(directory, '../../krazy_grid_world/maze_data/mazes_10k_20x20.npz'),
    "MazeTest-v0": os.path.join(directory, '../../krazy_grid_world/maze_data/mazes_test_10k_20x20.npz')
}
for env_id, path in MAZES.items():
    register(
        env_id,
        entry_point="custom_vendor.maze_env:MazeEnv",
        kwargs=dict(path=path, batch_size=1, n_episodes=1, episode_horizon=12),
        max_episode_steps=12,
        reward_threshold=50.0
    )
