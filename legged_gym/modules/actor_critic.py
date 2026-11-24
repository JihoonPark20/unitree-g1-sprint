"""256-unit feature MLP -> 64-unit LSTM -> Gaussian actor / value critic."""

import torch
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.modules import RNN
from rsl_rl.utils import unpad_trajectories
from torch import nn


class SprintRecurrentModel(RNNModel):
    def __init__(self, *args, feature_dim=256, rnn_hidden_dim=64, **kwargs):
        linear_head = not kwargs.get("hidden_dims", [])
        if linear_head:
            kwargs["hidden_dims"] = [rnn_hidden_dim]  # Upstream MLP requires a hidden layer.
        super().__init__(*args, rnn_hidden_dim=rnn_hidden_dim, **kwargs)
        if linear_head:
            self.mlp = nn.Sequential(nn.Linear(rnn_hidden_dim, self.mlp[-1].out_features))
            if self.distribution is not None:
                self.distribution.init_mlp_weights(self.mlp)
        self.features = nn.Sequential(nn.Linear(self.obs_dim, feature_dim), nn.ELU())
        self.rnn = RNN(feature_dim, rnn_hidden_dim, 1, "lstm")

    def get_latent(self, obs, masks=None, hidden_state=None):
        features = self.features(MLPModel.get_latent(self, obs))
        return self.rnn(features, masks, hidden_state).squeeze(0)

    def as_jit(self):
        raise NotImplementedError("Use the saved SprintRecurrentModel with its feature encoder and recurrent state")

    def as_onnx(self):
        raise NotImplementedError("An export wrapper must include the feature encoder and LSTM state")


class IntegratedMean:
    """Accumulate bounded Gaussian means in action units, once per policy step."""

    replay_key = "previous_clean_mean"

    def __init__(self, *args, action_limits, increment_kappa, **kwargs):
        super().__init__(*args, **kwargs)
        limits = torch.tensor(action_limits, dtype=torch.float32)
        self.register_buffer("mean_limit", limits, persistent=False)
        self.register_buffer("increment_limit", limits * increment_kappa, persistent=False)
        self.register_buffer("live_mean", None, persistent=False)
        self.register_buffer("rollout_previous_mean", None, persistent=False)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        delta = super().forward(obs, masks, hidden_state, stochastic_output=False)
        replay = self.replay_key in obs.keys()
        if replay:
            previous = obs[self.replay_key]
            if masks is not None:
                previous = unpad_trajectories(previous, masks)
        else:
            if masks is not None:
                raise ValueError("Integrated-mean PPO batches must contain the rollout's clean previous mean")
            previous = torch.zeros_like(delta) if self.live_mean is None else self.live_mean
        increment = delta.clamp(-self.increment_limit, self.increment_limit)
        mean = (previous + increment).clamp(-self.mean_limit, self.mean_limit)
        if not replay:
            self.rollout_previous_mean = previous
            self.live_mean = mean.detach()
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def reset(self, dones=None, hidden_state=None):
        super().reset(dones, hidden_state)
        if dones is None:
            self.live_mean = self.rollout_previous_mean = None
        elif self.live_mean is not None:
            self.live_mean = self.live_mean.masked_fill(dones.bool().reshape(-1, 1), 0)

    def as_jit(self):
        raise NotImplementedError("Export requires the previous clean mean as recurrent state")

    def as_onnx(self):
        raise NotImplementedError("Export requires the previous clean mean as recurrent state")


class IntegratedMeanLSTM(IntegratedMean, SprintRecurrentModel):
    """Recurrent SPRINT actor with integrated mean."""


class IntegratedMeanMLP(IntegratedMean, MLPModel):
    """Feedforward encoder with integrated mean and shuffled PPO replay."""
