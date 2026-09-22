"""QR-DQN con selección Double-DQN para reducir sobreestimación."""
from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import QRDQN
from sb3_contrib.common.utils import quantile_huber_loss


class DoubleQRDQN(QRDQN):
    """Usa la red online para elegir y la red objetivo para evaluar la acción.

    QR-DQN estándar aprende una distribución de retornos por cuantiles. Esta
    variante conserva esa pérdida y separa selección/evaluación como Double DQN.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )
            with th.no_grad():
                # Double DQN: argmax con online; cuantiles con target.
                online_next = self.quantile_net(replay_data.next_observations)
                greedy_actions = online_next.mean(dim=1).argmax(
                    dim=1, keepdim=True
                )
                greedy_actions = greedy_actions[:, None, :].expand(
                    batch_size, self.n_quantiles, 1
                )
                target_next = self.quantile_net_target(
                    replay_data.next_observations
                )
                target_next = target_next.gather(
                    dim=2, index=greedy_actions
                ).squeeze(dim=2)
                target_quantiles = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * target_next
                )

            current = self.quantile_net(replay_data.observations)
            actions = replay_data.actions[..., None].long().expand(
                batch_size, self.n_quantiles, 1
            )
            current = th.gather(current, dim=2, index=actions).squeeze(dim=2)
            loss = quantile_huber_loss(
                current, target_quantiles, sum_over_quantiles=True
            )
            losses.append(float(loss.item()))
            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

