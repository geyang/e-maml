import gym
gym.logger.set_level(40)  # set logging level to avoid annoying warning.


def __getattribute__(self, attr_name):
    try:
        return object.__getattribute__(self, attr_name)
    except AttributeError:
        return object.__getattribute__(self.env, attr_name)


gym.core.Wrapper.__getattribute__ = __getattribute__

### subproc_vec_env patches
import numpy as np
import baselines.common.vec_env.subproc_vec_env as subproc


def __getattr__(self, attr_name):
    if attr_name in dir(self):
        return object.__getattribute__(self, attr_name)
    else:
        def remote_exec(*args, **kwargs):
            for remote in self.remotes:
                remote.send((attr_name, dict(args=args, kwargs=kwargs)))
            return np.stack([remote.recv() for remote in self.remotes])

        return remote_exec


subproc.SubprocVecEnv.__getattr__ = __getattr__


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            # todo: to distinguish between a functional call and a getitem, this needs some more thought
            remote.send(getattr(env, cmd)(*data['args'], **data['kwargs']))


subproc.worker = worker

from gym.envs.mujoco.mujoco_env import MujocoEnv


def close(self):
    if self.viewer is not None:
        try:
            self.viewer.finish()
        except:
            pass
        self.viewer = None


MujocoEnv.close = close

from baselines.common import distributions


def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
    import tensorflow as tf
    import numpy as np
    import baselines.common.tf_util as U
    from baselines.a2c.utils import fc
    mean = fc(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
    logstd = tf.get_variable(name='logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
    return self.pdfromflat(pdparam), mean


distributions.DiagGaussianPdType.pdfromlatent = pdfromlatent
