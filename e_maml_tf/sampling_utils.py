from numbers import Number


def is_scalar(n):
    return (hasattr(n, 'shape') and len(n.shape) < 1) or isinstance(n, Number)


def batchify(paths, batch_size, n, shuffle):
    """

    :param paths:
    :param batch_size:
    :param n: length of the
    :param shuffle: boolean flag to shuffle the batch.
    :return:
    """
    import numpy as np
    shuffled_inds = np.random.randn(n).argsort()
    for i in range(n // batch_size):
        start = i * batch_size
        end = start + batch_size
        yield {
            k: v if is_scalar(v) else v[shuffled_inds[start:end] if shuffle else range(start, end)]
            for k, v in paths.items()
        }
