# E-MAML Implementation

This repo contains the full implementation of the E-MAML algorithm from the paper
[*Some Considerations on Learning to Explore via Meta-Reinforcement Learning*][NIPS_link]

[NIPS_link]: https://papers.nips.cc/paper/8140-the-importance-of-sampling-inmeta-reinforcement-learning

## Structure of This Codebase

The main implementation is contained in the `e_maml_tf` directory. Inside the `e_maml_experiments` 
directory we provide a light weight half-cheetah baseline for verification. The original
KrazyWorld codebase is not opensourced. So we implemented a new KrazyWorld environment. To run E-MAML
on this new KrazyWorld, you need to add a thin adaptor following the convention in `custom_vendor`
and `sampler.py`.

:point_right: [`KrazyWorld` github repo][KrazyWorld]

[KrazyWorld]: https://github.com/bstadie/krazyworld.git

# Getting Started:

1. Setup conda environment with python 3.6.4 or above. (this is required for all of the `f-string` literals.)
2. if on mac, run `brew install mpich`. this is the MPI version that `baseline` and `mpi4py` relies on.
3. run `pip install -e .`. If the `mpi4py` installation fails, try `pip install mpi4py` in a new terminal session.
4. if `mujoco-py` complains (which fails the installation), make sure you have installed mujoco and have a working license key.
5. If not, you should download mujoco for your environment and place the license key `mjkey.txt` under `~/.mujoco/`.
6. Distributed Setup: Add a file `.yours` inside `e_maml_experiments` that contains the following content:

    ```yaml
    username: <your-id>
    project: e_maml
    logging_server: http://<your-ml-logger-logging-server>:8081
    ```
   
   If you are not using a distributed logging setup, you can leave the logging_server to `none` or 
   leave it empty. In that case it would be logged to you `~/ml-logger-outputs` directory.
   
# Cite

To cite E-MAML please use

```bibtex
@article{stadie2018e-maml,
  title={Some considerations on learning to explore via meta-reinforcement learning},
  author={Stadie, Bradly C and Yang, Ge and Houthooft, Rein and Chen, Xi and Duan, Yan and Wu, Yuhuai and Abbeel, Pieter and Sutskever, Ilya},
  journal={arXiv preprint arXiv:1803.01118},
  year={2018}
}
```
