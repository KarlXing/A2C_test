import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
from visualize import visdom_plot
from tensorboardX import SummaryWriter
import pickle
import datetime
from utils import update_base_reward

#####################################
# prepare

now = datetime.datetime.now().isoformat()

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.log_dir = args.log_dir + now

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

# eval_log_dir = args.log_dir + "_eval/" 

# try:
#     os.makedirs(eval_log_dir)
# except OSError:
#     files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
#     for f in files:
#         os.remove(f)

save_path = os.path.join(args.save_dir, args.algo)
try:
    os.makedirs(save_path)
except OSError:
    pass

reward_path = os.path.join(args.reward_dir, args.algo)
try:
    os.makedirs(reward_path)
except OSError:
    pass


######################################
# main

def main():
    writer = SummaryWriter()
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    best_score = 0

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    if args.reward_mode == 0:
        clip_rewards = True
    else:
        clip_rewards = False
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, 4, clip_rewards, args.track_primitive_reward)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space, args.activation,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    # initiate env and storage rollout
    obs = envs.reset()
    obs = obs/255
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # necessary variabels
    episode_rewards = deque(maxlen=10) # store last 10 episode rewards
    g_step = 0 # global step
    # reward_history = set() # record reward history (after reward rescaling)
    # primitive_reward_history = set() # record original history (before reward rescaling)
    # min_abs_reward = float('inf') # used in reward rescaling mode 2, work as a base
    masks_device = torch.ones(args.num_processes, 1).to(device)  # mask on gpu
    reward_count = 0 # for reward density calculation
    reward_start_step = 0 # for reward density calculation
    base_reward = None
    rewards = {}
    insert_entropy = torch.ones(args.num_processes, 1)  # entropys inserte into rollout

    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            g_step += 1
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, entropy, f_a = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            obs = obs/255
            for r in reward:
                if r.item() not in rewards:
                    rewards[r.item()] = 0
                else:
                    rewards[r.item()] += 1

            # reward rescaling
            if args.reward_mode == 2:
                base_reward, ratio, update = update_base_reward(reward, base_reward)
                if base_reward is not None:
                    reward = reward/base_reward
                if ratio != 1:
                    actor_critic.base.update_critic(ratio)
                if update:
                    writer.add_scalar('base/new_base_reward', base_reward, g_step)


            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

            if args.log_evaluation:
                writer.add_scalar('analysis/entropy', entropy.mean().item(), g_step)
                if args.track_reward_density:   # track reward density, based on 0th process
                    reward_count += (reward[0] != 0)
                    if 'episode' in infos[0].keys():
                        writer.add_scalar('analysis/reward_density', reward_count/(g_step - reward_start_step), g_step)
                        reward_count = 0
                        reward_start_step = g_step
                if args.track_primitive_reward:   # track primitive reward (before rescaling)
                    for info in infos:
                        if 'new_reward' in info:
                            new_rewards  = info['new_reward'] - primitive_reward_history
                            if len(new_rewards) > 0:
                                print('new primitive rewards: ', new_rewards, ' time: ', g_step)
                                primitive_reward_history =  primitive_reward_history.union(info['new_reward'])

            for idx in range(len(infos)):
                info = infos[idx]
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    steps_done = g_step*args.num_processes + idx
                    writer.add_scalar('data/reward', info['episode']['r'], steps_done)
                    mean_rewards = np.mean(episode_rewards)
                    writer.add_scalar('data/avg_reward', mean_rewards, steps_done)
                    if mean_rewards > best_score:
                        best_score = mean_rewards
                        save_model = actor_critic
                        if args.cuda:
                            save_model = copy.deepcopy(actor_critic).cpu()
                        torch.save(save_model, os.path.join(save_path, args.env_name + now + ".pt"))
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, insert_entropy)

        with torch.no_grad():
            masks_device.copy_(masks)
            next_value = actor_critic.get_value(obs, recurrent_hidden_states, masks_device)

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy, value = agent.update(rollouts, args.modulation)

        if args.track_value_loss:
            writer.add_scalar('analysis/value_loss', value_loss, j)
            writer.add_scalar('analysis/value', value, j)
            writer.add_scalar('analysis/loss_ratio', value_loss/value, j)

        rollouts.after_update()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    with open(os.path.join(reward_path, args.env_name + now + '_reward.dic'), 'wb') as f:
        pickle.dump(rewards, f)

if __name__ == "__main__":
    main()
