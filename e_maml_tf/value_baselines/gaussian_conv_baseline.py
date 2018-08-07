import numpy as np

from .base import Baseline
from rllab.regressors.gaussian_conv_regressor import GaussianConvRegressor


class GaussianConvBaseline(Baseline):
    def __init__(self, env_spec, regressor_args=None, ):
        super().__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianConvRegressor(
            input_shape=env_spec.observation_space.shape,
            output_dim=1,
            name="vf",
            **regressor_args
        )

    def fit(self, paths):
        observations = np.concatenate([p["obs"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    def fit_by_samples_data(self, samples_data):
        observations = samples_data["obs"]
        returns = samples_data["returns"]
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    def predict(self, path):
        return self._regressor.predict(path["obs"]).flatten()

    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
