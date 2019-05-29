import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards_ex = torch.zeros(num_steps, num_processes, 1)
        self.rewards_in = torch.zeros(num_steps, num_processes, 1)
        self.value_preds_ex = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds_in = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns_ex = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns_in = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0
        self.entropys = torch.zeros(num_steps, num_processes, 1)
        self.lr = torch.ones(num_steps, num_processes, 1)

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards_ex = self.rewards_ex.to(device)
        self.rewards_in = self.rewards_in.to(device)
        self.value_preds_ex = self.value_preds_ex.to(device)
        self.value_preds_in = self.value_preds_in.to(device)
        self.returns_ex = self.returns_ex.to(device)
        self.returns_in = self.returns_in.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.entropys = self.entropys.to(device)
        self.lr = self.lr.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds_ex, value_preds_in, rewards_ex, rewards_in, masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds_ex[self.step].copy_(value_preds_ex)
        self.value_preds_in[self.step].copy_(value_preds_in)
        self.rewards_ex[self.step].copy_(rewards_ex)
        self.rewards_in[self.step].copy_(rewards_in)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def insert_lr(self, lr):
        self.lr.copy_(lr)

    def insert_obs(self, obs):
        self.obs[self.step + 1].copy_(obs)
        self.step = (self.step + 1) % self.num_steps

    def clean(self):   # called when initialization is done
        self.obs = self.obs * 0
        self.step  = 0

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value_ex, next_value_in, use_gae, gamma, tau):
        # todo: support use_gae (random network distillation)
        assert(use_gae == False)
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns_ex[-1] = next_value_ex
            self.returns_in[-1] = next_value_in
            for step in reversed(range(self.rewards_ex.size(0))):
                self.returns_ex[step] = self.returns_ex[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards_ex[step]
            for step in reversed(range(self.rewards_in.size(0))):
                self.returns_in[step] = self.returns_in[step + 1] * gamma + self.rewards_in[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards_ex.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_ex_batch = self.value_preds_ex[:-1].view(-1, 1)[indices]
            value_preds_in_batch = self.value_preds_in[:-1].view(-1, 1)[indices]
            return_ex_batch = self.returns_ex[:-1].view(-1, 1)[indices]
            return_in_batch = self.returns_in[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_ex_batch, value_preds_in_batch, return_ex_batch, return_in_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
