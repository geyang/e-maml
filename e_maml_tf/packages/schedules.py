from collections import Generator
import inspect
from typing import Callable


class Schedule(Generator):
    def __init__(self, schedule_fn: Callable):
        assert callable(schedule_fn), 'need to pass in a real callable.'
        self.schedule_fn = schedule_fn
        source = inspect.getsource(self.schedule_fn)
        self.repr = "Schedule:\n" + source.rstrip() if len(source.split('\n')[1]) > 1 else source.rstrip()

    def send(self, epoch_ind):
        return self.schedule_fn(epoch_ind)

    def throw(self, *args):
        raise StopIteration(*args)

    def __repr__(self):
        return self.repr

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    s = Schedule(lambda i: 10 if i < 10 else 50)
    print(s)


    def longer_schedule(i):
        if i < 10:
            return 5
        elif i < 40:
            return 10
        return 50


    s = Schedule(longer_schedule)
    print(s)

    assert s.send(1) == 5
    assert s.send(20) == 10
    assert s.send(50) == 50


def dilated_delta(n, k):
    """Dilated Delta Schedule function

    returns a dilated delta function, starting with 0 and increasing, with double
    of the duty cycle after each cycle.

    :param n: total number of steps
    :param k: number of cycles
    :return: value between 0 and 1, in floats
    """
    import numpy as np
    import math

    ints = 2 ** np.arange(k)
    ends = ints * 2 - 1
    si = np.concatenate([[(e - i, i)] * i for i, e in zip(ints, ends)])
    schedule_ratio = n / len(si)

    def dilated_delta_fn(ep):
        i = math.floor(ep / schedule_ratio)
        s, i = si[i]
        return (ep - schedule_ratio * s) / (schedule_ratio * i)

    return dilated_delta_fn


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    delta_anneal_fn = dilated_delta(1500, 5)
    betas = [delta_anneal_fn(i) for i in range(1500)]

    plt.figure(figsize=(4, 2))
    plt.title('dilated delta factory')
    plt.plot(betas)
    plt.show()


class DeltaAneal(Schedule):
    def __init__(self, min, max, n, k):
        """Delta Anneal Scheduler,

        Starting from max, goes down linearly to min, then repeat with 

        :param min: minimum of the parameter, to which the schedule converges to
        :param max: maximum of the parameter, that the schedule starts with
        :param n: the total number of epochs for this schedule
        :param k: the number of dilated cycles.
        :return: A dilated delta annealing schedule generator g, call g.send(ep) for the parameter value.
        """
        delta_fn = dilated_delta(n, k)
        super().__init__(lambda ep: max - (max - min) * delta_fn(ep))
        self.repr = f"DeltaAnneal(min={min}, max={max}, n={n}, k={k})"


if __name__ == "__main__":
    s = DeltaAneal(0.04, 0.1, 1500, 4)
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 2))
    plt.title(f'{s}')
    plt.plot([s.send(x) for x in range(1500)])
    plt.ylim(-0.1, 0.2)
    plt.show()


class CosineAnneal(Schedule):
    def __init__(self, min, max, n, k):
        """Cosine Anneal Scheduler,

        Starting from max, goes down as a cosine function to min, then repeat with

        :param min: minimum of the parameter, to which the schedule converges to
        :param max: maximum of the parameter, that the schedule starts with
        :param n: the total number of epochs for this schedule
        :param k: the number of dilated cycles.
        :return: A dilated delta annealing schedule generator g, call g.send(ep) for the parameter value.
        """
        import numpy as np
        delta_fn = dilated_delta(n, k)
        super().__init__(lambda ep: min + (max - min) * 0.5 * (1 + np.cos(np.pi * delta_fn(ep))))
        self.repr = f"CosineAnneal(min={min}, max={max}, n={n}, k={k})"


if __name__ == "__main__":
    s = CosineAnneal(0.04, 0.1, 1500, 4)
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 2))
    plt.title(f'{s}')
    plt.plot([s.send(x) for x in range(1500)])
    plt.ylim(-0.1, 0.2)
    plt.show()


# test that the instance detection still works
if __name__ == "__main__":
    s = CosineAnneal(0.04, 0.1, 1500, 4)
    assert isinstance(s, Schedule), "CosineAnneal is an instance of Schedule"
    print('passed the `isinstance` test!!')
