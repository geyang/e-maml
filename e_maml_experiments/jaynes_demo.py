"""A simple demo for launching ML jobs with Jaynes.
"""


def train_fn(some_variable=0):
    import tensorflow as tf
    print(f"tensorflow version: {tf.__version__}")

    print('training is happening!')
    print("some_variable is", some_variable)


if __name__ == "__main__":
    import jaynes

    jaynes.config('default')
    jaynes.run(train_fn, some_variable=5)

    jaynes.listen(timeout=60)
