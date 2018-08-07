"""Minimal Exasmple for E-MAML running on HalfCheetahGoalDir-v0
"""
from e_maml_tf.config import RUN, G, Reporting, DEBUG
from e_maml_tf.train import run_e_maml

if __name__ == '__main__':

    G.env_name = "HalfCheetahGoalDir-v0"
    G.n_tasks = 20
    G.n_graphs = 1
    G.n_grad_steps = 5
    G.meta_n_grad_steps = 1

    # to debug this locally
    run_e_maml()

    # to launch with Jaynes on a SLURM cluster
    import jaynes
    jaynes.config('default')
    jaynes.run(run_e_maml, _G=vars(G))
