"""PPO timeout bootstrap uses the actual terminal state, preserving LSTM state."""

import torch
from rsl_rl.algorithms import PPO


class SprintPPO(PPO):
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
