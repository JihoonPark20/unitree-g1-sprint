"""PPO timeout bootstrap uses the actual terminal state, preserving LSTM state."""

import torch
from rsl_rl.algorithms import PPO


class SprintPPO(PPO):
    def update(self):
        # Forward clamping alone leaves zero gradients for std parameters that
        # drift outside the interval. Projection keeps exploration trainable.
        self._project_action_std()
        hook = self.optimizer.register_step_post_hook(lambda *_: self._project_action_std())
        try:
            return super().update()
        finally:
            hook.remove()

    @torch.no_grad()
    def _project_action_std(self):
        distribution = self.actor.distribution
        if distribution.std_type == "scalar":
            parameter, bounds = distribution.std_param, distribution.std_range
        elif distribution.std_type == "log":
            parameter, bounds = distribution.log_std_param, distribution.log_std_range
        else:
            raise ValueError("Unsupported action standard deviation parameterization")
        # Some torch versions also give clamp zero gradient at the boundary.
        # Stay one dtype-scale epsilon inside it, including in log space.
        lo, hi = bounds
        margin = min(torch.finfo(parameter.dtype).eps * max(1.0, abs(lo), abs(hi)), (hi - lo) / 4)
        parameter.clamp_(lo + margin, hi - margin)

    def process_env_step(self, obs, rewards, dones, extras):
        timeout = extras.get("time_outs")
        if timeout is not None and timeout.any():
            if "terminal_observations" not in extras:
                raise ValueError("Time limits require pre-reset terminal observations")
            terminal = obs.clone().to(self.device)
            ids = extras["terminal_env_ids"].to(self.device)
            terminal[ids] = extras["terminal_observations"].to(self.device)
            # critic has consumed o_t. Evaluate o_{t+1} only for timeout targets,
            # then restore its state so the next rollout call consumes it once.
            hidden = self.critic.get_hidden_state()
            with torch.no_grad():
                values = self.critic(terminal).squeeze(-1)
            self.critic.reset(hidden_state=hidden)
            mask = timeout.to(device=self.device, dtype=rewards.dtype)
            # rsl-rl adds gamma*V(o_t) below; replace that with V(terminal).
            rewards = rewards + self.gamma * (values - self.transition.values.squeeze(-1)) * mask
        super().process_env_step(obs, rewards, dones, extras)
