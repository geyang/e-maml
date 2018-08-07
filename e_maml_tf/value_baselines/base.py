class Baseline(object):
    @property
    def algorithm_parallelized(self):
        return False

    def get_param_values(self):
        raise NotImplementedError

    def set_param_values(self, val):
        raise NotImplementedError

    def fit(self, obs, rewards, returns):
        raise NotImplementedError

    def predict(self, obs, rewards):
        raise NotImplementedError

    @classmethod
    def add_args(cls, parser):
        pass

    @classmethod
    def new_from_args(cls, args, mdp):
        pass

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass
