import mock
import tensorflow as tf
from typing import Callable, Any, List, TypeVar


def probe_var(*variables):
    return tf.get_default_session().run(variables)


def as_dict(c):
    return {k: v for k, v in vars(c).items() if k[0] != "_"}


def var_like(var, trainable=False):
    name, dtype, shape = var.name, var.dtype, tuple(var.get_shape().as_list())
    new_name = name.split(':')[0]
    # note: assuming that you are using a variable scope for this declaration.
    new_var = tf.Variable(initial_value=tf.zeros(shape, dtype), name=new_name)
    # print(f"declaring variable like {name} w/ new name: {new_var.name}")
    return new_var


def placeholders_from_variables(var, name=None):
    """Returns a nested collection of TensorFlow placeholders that match shapes
    and dtypes of the given nested collection of variables.
    Arguments:
    ----------
        var: Nested collection of variables.
        name: Placeholder name.
    Returns:
    --------
        Nested collection (same structure as `var`) of TensorFlow placeholders.
    """
    if isinstance(var, list) or isinstance(var, tuple):
        result = [placeholders_from_variables(v, name) for v in var]
        if isinstance(var, tuple):
            return tuple(result)
        return result
    else:
        dtype, shape = var.dtype, tuple(var.get_shape().as_list())
        return tf.placeholder(dtype=dtype, shape=shape, name=name)


def wrap_variable_creation(func, custom_getter):
    """Provides a custom getter for all variable creations."""
    original_get_variable = tf.get_variable

    def custom_get_variable(*args, **kwargs):
        if hasattr(kwargs, "custom_getter"):
            raise AttributeError("Custom getters are not supported for optimizee variables.")
        return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

    # Mock the get_variable method.
    with mock.patch("tensorflow.get_variable", custom_get_variable):
        return func()


def get_var_name(string):
    return string.split(':')[0]


def var_map(variables, root_scope_name):
    """
    only returns those that starts with the root_scope_name.

    :param variables:
    :param root_scope_name:
    :return:
    """
    return {get_var_name(v.name)[len(root_scope_name):]: v for v in variables if v.name.startswith(root_scope_name)}


def get_scope_name():
    return tf.get_default_graph().get_name_scope()


def stem(n, k=1):
    """
    Allow using k > 1 to leave a longer segment of the bread crum

    Example Variable(output Tensor) Names:
    ```
        runner/input_bias:0
        runner/MlpPolicy/pi_fc1/w:0
        runner/MlpPolicy/pi_fc1/b:0
        runner/MlpPolicy/pi_fc2/w:0
    ```

    stem(tensor.name, 2) should give us

    ```
        runner/input_bias
        pi_fc1/w
        pi_fc1/b
        pi_fc2/w
    ```


    :param n:
    :param k:
    :return:
    """
    return "/".join(n.split(":")[0].split('/')[-k:])


T = TypeVar('T')


def make_with_custom_variables(func: Callable[[Any], T], variable_map, root_name_space="") -> T:
    """Calls func and replaces any trainable variables.
    This returns the output of func, but whenever `get_variable` is called it
    will replace any trainable variables with the tensors in `variables`, in the
    same order. Non-trainable variables will re-use any variables already
    created.
    Arguments:
    ----------
        func: Function to be called.
        variables: A list of tensors replacing the trainable variables.
    Returns:
    --------
        The return value of func is returned.
    """

    def custom_getter(getter, name, **kwargs):
        nonlocal variable_map
        postfix = name[len(root_name_space):]
        return variable_map[postfix]

    return wrap_variable_creation(func, custom_getter)


# noinspection PyPep8Naming
class defaultlist():
    """allow using -1, -2 index to query from the end of the list, which is not possible with `defaultdict`. """

    def __init__(self, default_factory):
        self.data = list()
        self.default_factory = default_factory if callable(default_factory) else lambda: default_factory

    def __setitem__(self, key, value):
        try:
            self.data[key] = value
        except IndexError:
            self.data.extend([self.default_factory()] * (key + 1 - len(self.data)))
            self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]

    def __setstate__(self, state):
        raise NotImplementedError('need to be implemented for remote execution.')

    def __getstate__(self):
        raise NotImplementedError('need to be implemented for remote execution.')


class Cache:
    def __init__(self, variables):
        """
        creates a variable flip-flop in-memory.

        :param variables:
        :return: save_op, load_op, cache array
        """
        self.cache = [var_like(v) for v in variables]
        self.save = tf.group(*[c.assign(tf.stop_gradient(v)) for c, v in zip(self.cache, variables)])
        self.load = tf.group(*[v.assign(tf.stop_gradient(c)) for c, v in zip(self.cache, variables)])


# goal: try to add the gradients without going through python.
class GradientSum:
    def __init__(self, variables, grad_inputs):
        """k is the number of gradients you want to sum.
        zero this gradient op once every meta iteration. """
        self.cache = [var_like(v) for v in variables]
        # call set before calling add op, faster than zeroing out the cache.
        self.set_op = tf.group(*[c.assign(tf.stop_gradient(g)) for c, g in zip(self.cache, grad_inputs)])
        self.add_op = tf.group(*[c.assign_add(tf.stop_gradient(g)) for c, g in zip(self.cache, grad_inputs)])


def flatten(arr):
    """swap and then flatten axes 0 and 1"""
    n_steps, n_envs, *_ = arr.shape
    return arr.swapaxes(0, 1).reshape(n_steps * n_envs, *_)
