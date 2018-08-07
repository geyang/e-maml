import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


# make synchronous interface for get call

def worker(remote, parent_remote, env):
    parent_remote.close()
    env = env.x() if hasattr(env, 'x') else env()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'get':
                remote.send(getattr(env, data))
            elif cmd == 'close':
                remote.close()
                break  # this terminates the process.
            else:
                data = data or dict()
                args = data.get('args', tuple())
                kwargs = data.get('kwargs', dict())
                _ = getattr(env, cmd)(*args, **kwargs)
                remote.send(_)

        except EOFError as e:  # process has ended from inside
            break  # this terminates the process
        except BaseException as e:
            print(e)
            break


class SubprocVecEnv:
    reset_on_done = True

    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.first = self.remotes[0]
        self.first.send(('get', 'action_space'))
        self.action_space = self.first.recv()
        self.first.send(('get', 'observation_space'))
        self.observation_space = self.first.recv()
        self.first.send(('get', 'spec'))
        self.spec = self.first.recv()

    def fork(self, n):
        from copy import copy
        _self = copy(self)
        _self.remotes = _self.remotes[:n]
        return _self

    def call_sync(self, fn_name, *args, **kwargs):
        _ = fn_name, dict(args=args, kwargs=kwargs)
        for remote in self.remotes:
            remote.send(_)
        try:
            return np.stack([remote.recv() for remote in self.remotes])
        except EOFError as e:
            raise RuntimeError('Unknown Error has occurred with the environment.') from e

    def get(self, key):
        raise NotImplementedError('need to decide for self.first or all.')

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        obs, rews, dones, infos = zip(*[remote.recv() for remote in self.remotes])
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def render(self, *args, **kwargs):
        return self.call_sync('render', *args, **kwargs)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        return self.call_sync('reset')

    def sample_task(self, *args, **kwargs):
        return self.call_sync('sample_task', *args, **kwargs)

    def reset_task(self):
        self.call_sync('reset_task')

    def close(self):
        """looks bad: mix sync and async handling."""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def first_call_sync(self, fn_name, *args, **kwargs):
        self.first.send((fn_name, dict(args=args, kwargs=kwargs)))
        return self.first.recv()


if __name__ == "__main__":
    def make_env():
        import gym
        return gym.make('Reacher-v2')


    parallel_envs = SubprocVecEnv([make_env for i in range(6)])
    obs = parallel_envs.reset()
    assert len(obs) == 6, "the original should have 6 envs"

    render_envs = parallel_envs.fork(4)
    # note: here we test the `fork` method, useful for selectiong a sub-batch for rendering purposes.
    obs = render_envs.reset()
    assert len(obs) == 4, "the forked env should have only 4 envs."

    print('test complete.')
