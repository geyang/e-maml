from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    dependencies = f.read()

setup(name='leaf',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "mpi4py",
          "scipy",
          "pandas",
          "gym",
          "baselines",
          "tqdm",
          "params-proto",
          "tensorflow-gpu",
          "ml-logger",
          "moleskin",
          "jaynes",
          "pyyaml",
          "waterbear",
          "dill",
          "mock",
          "mujoco-py",
      ],
      description='E-MAML, and RL-MAML baseline implemented in Tensorflow v1',
      author='Ge Yang',
      url='https://github.com/episodeyang/e-maml',
      author_email='ge.ike.yang@gmail.com',
      version='0.0.1')
