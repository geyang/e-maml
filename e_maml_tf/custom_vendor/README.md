# Custom Vendor (Patches)

This module patches OpenAI `gym` and `al_algs`. Specifically, it

- adds various gym tasks that are used in the `maml-rl` experiments
    - `HalfCheetahGoalVel-v0`: This is the Cheetah task with a velocity goal.
    
- patches the `al_algs` `Wrapper` class, which was implemented disregarding wrappee method passing.
- same applying to `SubprocVecEnv` and it's `worker` method.

The reason to monkey patch is so that these modifications can stay at one place instead of buried in the forked source code of these large libraries. This makes it easier to merge back upstream.