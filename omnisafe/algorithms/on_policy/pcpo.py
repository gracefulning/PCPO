# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modifications made by maintainers.
# Summary: this file has been modified from the original OmniSafe version
#          to implement our algorithm.

"""Implementation of PCPO algorithm."""

import torch

from torch.distributions import Distribution

import math

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)

@registry.register
class PCPO(TRPO):
    def _init(self) -> None:
        """Initialize the PCPO specific model.
        """
        super()._init()
        
    def _init_log(self) -> None:
        """Log the PCPO specific information.
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        
    def _update(self) -> None:
        super()._update()

   
    def _loss_pi(self, obs, act, logp, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Compute the policy loss with log-barrier method."""

        distribution = self._actor_critic.actor(obs)
        if isinstance(distribution, tuple):
            distribution, mask = distribution
            logp_ = self._actor_critic.actor.log_prob(act, mask)
        else:
            logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        entropy = distribution.entropy().mean().item()
        std = self._actor_critic.actor.std

        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        d = self._cfgs.lagrange_cfgs.cost_limit
        
        gc = ((Jc - d) + (ratio * adv_c).mean() / (1-self._cfgs.algo_cfgs.gamma))
       
        tau = self._cfgs.lagrange_cfgs.penalty
        
        threshold = -1 / tau**2

        if gc < threshold:
            varphi = -1/tau * torch.log(-gc)
        else:
            varphi = tau * gc
        
        loss = -(ratio * adv_r).mean() + varphi

        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

    def _update_actor(  
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """
        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        loss = self._loss_pi(obs, act, logp, adv_r, adv_c)

        loss_before = distributed.dist_avg(loss).item()
        p_dist = self._actor_critic.actor(obs)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'
        total_steps = 15
        step_direction, accept_step = self._search_step_size(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            loss_before=loss_before,
            total_steps=total_steps,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss = self._loss_pi(obs, act, logp, adv_r, adv_c)

        self._logger.store(
            {
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
                'Misc/AcceptanceStep': accept_step,
            },
        )
    

    def _search_step_size(
        self,
        step_direction: torch.Tensor,
        grads: torch.Tensor,
        p_dist: Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        loss_before: float,
        total_steps: int = 15,
        decay: float = 0.8,
    ):#-> tuple[torch.Tensor, int]:
        """
        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            logp (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            adv_c (torch.Tensor): The cost advantage.
            loss_pi_before (float): The loss of the policy before the update.
            total_steps (int, optional): The total steps to search. Defaults to 15.
            decay (float, optional): The decay rate of the step size. Defaults to 0.8.
        Returns:
            The tuple of final update direction and acceptance step size.
        """
        # How far to go in a single update
        step_frac = 1.0
        # Get old parameterized policy expression
        theta_old = get_flat_params_from(self._actor_critic.actor)
        # Change expected objective function gradient = expected_imrpove best this moment
        expected_improve = grads.dot(step_direction)

        final_kl = 0.0

        # While not within_trust_region and not out of total_steps:
        for step in range(total_steps):
            # update theta params
            new_theta = theta_old + step_frac * step_direction
            # set new params as params of net
            set_param_values_to_model(self._actor_critic.actor, new_theta)

            with torch.no_grad():
                loss = self._loss_pi(obs, act, logp, adv_r, adv_c)
                # compute KL distance between new and old policy
                q_dist = self._actor_critic.actor(obs)
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
                kl = distributed.dist_avg(kl).mean().item()
            # real loss improve: old policy loss - new policy loss
            loss_improve = loss_before - loss.item()
            # average processes.... multi-processing style like: mpi_tools.mpi_avg(xxx)
            loss_improve = distributed.dist_avg(loss_improve)
            self._logger.log(f'Expected Improvement: {expected_improve} Actual: {loss_improve}')
            if not torch.isfinite(loss):
                self._logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self._logger.log('INFO: did not improve improve <0')
            elif kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                acceptance_step = step + 1
                self._logger.log(f'Accept step at i={acceptance_step}')
                final_kl = kl
                break
            step_frac *= decay
        else:
            self._logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        set_param_values_to_model(self._actor_critic.actor, theta_old)

        self._logger.store(
            {
                'Train/KL': final_kl,
            },
        )

        return step_frac * step_direction, acceptance_step
