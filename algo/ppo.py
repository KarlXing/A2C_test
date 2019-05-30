import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False,
                 rnd_loss_coef=0.05):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.rnd_loss_coef = rnd_loss_coef
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts, obs_rms):
        advantages = 2 * (rollouts.returns_ex[:-1] - rollouts.value_preds_ex[:-1]) + (rollouts.returns_in[:-1]  - rollouts.value_preds_in[:-1])
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        rnd_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_ex_batch, value_preds_in_batch, return_ex_batch, return_in_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values_ex, values_in, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_ex_clipped = value_preds_ex_batch + \
                        (values_ex - value_preds_ex_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses_ex = (values_ex - return_ex_batch).pow(2)
                    value_losses_ex_clipped = (value_pred_ex_clipped - return_ex_batch).pow(2)

                    value_pred_in_clipped = value_preds_in_batch + \
                        (values_in - value_preds_in_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses_in = (values_in - return_in_batch).pow(2)
                    value_losses_in_clipped = (value_pred_in_clipped - return_in_batch).pow(2)

                    value_loss = .5 * (torch.max(value_losses_in, value_losses_in_clipped).mean() + torch.max(value_losses_ex, value_losses_ex_clipped).mean())
                else:
                    value_loss = 0.5 * (F.mse_loss(return_ex_batch, values_ex) + F.mse_loss(return_in_batch, values_in))

                # random network loss
                rnd_obs = obs_batch[:,3:,:,:]
                rnd_obs = ((rnd_obs - obs_rms.mean) / torch.sqrt(obs_rms.var)).clamp(-5, 5)
                predict_feature, target_feature = self.actor_critic.rnd(rnd_obs)
                rnd_loss = F.mse_loss(predict_feature, target_feature.detach())

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef + rnd_loss * self.rnd_loss_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                rnd_loss_epoch += rnd_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        rnd_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, rnd_loss_epoch
