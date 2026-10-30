"""256-unit feature MLP -> 64-unit LSTM -> Gaussian actor / value critic."""

from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.modules import RNN
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
