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
from model import Policy as Policy_old
from model2 import Policy
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
    clip_rewards = True  #force clip rewards
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, 4, clip_rewards, args.track_primitive_reward)

    # load pretrained model
    if ("alien" in args.env_name.lower()):
        old_ac = torch.load("/pv/models/alien01.pt")
    else if ("pacman" in args.env_name.lower()):
        old_ac = torch.load("pv/models/mspacman01.pt")
    else if ("phoenix" in args.env_name.lower()):
        old_ac = torch.load("/pv/models/phoenix01.pt")
    else if ("beamrider" in args.env_name.lower()):
        old_ac = torch.load("/pv/models/beamrider01.pt")

    actor_critic = Policy(envs.observation_space.shape, envs.action_space, args.activation,
        base_kwargs={'recurrent': args.recurrent_policy})

    # load trained weights
    actor_critic.base.conv1.weight.data = old_ac.base.conv1.weight.data
    actor_critic.base.conv1.bias.data = old_ac.base.conv1.bias.data
    actor_critic.base.conv1_2.weight.data = old_ac.base.conv1.weight.data
    actor_critic.base.conv1_2.bias.data = old_ac.base.conv1.bias.data
    actor_critic.base.conv2.weight.data = old_ac.base.conv2.weight.data
    actor_critic.base.conv2.bias.data = old_ac.base.conv2.bias.data
    actor_critic.base.conv2_2.weight.data = old_ac.base.conv2.weight.data
    actor_critic.base.conv2_2.bias.data = old_ac.base.conv2.bias.data
    actor_critic.base.conv3.weight.data = old_ac.base.conv3.weight.data
    actor_critic.base.conv3.bias.data = old_ac.base.conv3.bias.data
    actor_critic.base.conv3_2.weight.data = old_ac.base.conv3.weight.data
    actor_critic.base.conv3_2.bias.data = old_ac.base.conv3.bias.data
    actor_critic.base.f.weight.data = old_ac.base.f.weight.data
    actor_critic.base.f.bias.data = old_ac.base.f.bias.data
    actor_critic.base.f_2.weight.data = old_ac.base.f.weight.data
    actor_critic.base.f_2.bias.data = old_ac.base.f.bias.data
    actor_critic.base.critic_linear.weight.data = old_ac.base.critic_linear.weight.data
    actor_critic.base.critic_linear.bias.data = old_ac.base.critic_linear.bias.data
    actor_critic.dist.linear.weight.data = old_ac.dist.linear.weight.data
    actor_critic.dist.linear.bias.data = old_ac.dist.linear.bias.data

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
    # rewards = {}
    insert_entropy = torch.ones(args.num_processes, 1)  # entropys inserte into rollout
    value_ratio = 1.0  # the ratio between actual state value and critic value
    running_mean_value = 0.0

    # initialize the running_mean_value
    for i in range(args.initiate_steps):
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, entropy, f_a = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
            obs, reward, done, infos = envs.step(action)
            obs = obs/255
            running_mean_value = running_mean_value * 0.99 + 0.01 * torch.mean(value).item()
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert_part(obs, recurrent_hidden_states, masks)

    rollouts.reset()
    have_done = 0
    for j in range(num_updates):
        if j == int((num_updates-1)*have_done):
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            torch.save(save_model, os.path.join(save_path, args.env_name + str(args.lr) + "_" + str(have_done) + ".pt"))
            print("have done: ", have_done)
            have_done += 0.1
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
            # for r in reward:
            #     if r.item() not in rewards:
            #         rewards[r.item()] = 0
            #     else:
            #         rewards[r.item()] += 1

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

        advantage, value_loss, action_loss, dist_entropy, value, critic_grad, actor_grad = agent.update(rollouts)

        if args.track_value_loss:
            writer.add_scalar('analysis/advantage', advantage, j)
            writer.add_scalar('analysis/value', value, j)
            writer.add_scalar('analysis/value_loss', value_loss, j)
            writer.add_scalar('analysis/action_loss', action_loss, j)

        if args.track_grad:
            # value of gradient and how much gradients are cancelled out
            # value
            abs_critic_grad = [torch.abs(grad) for grad in critic_grad]
            abs_actor_grad = [torch.abs(grad) for grad in actor_grad]
            writer.add_scalars('analysis/critic_grad',{'conv1': torch.mean(abs_critic_grad[0]).item()},j)
            writer.add_scalars('analysis/critic_grad',{'conv2': torch.mean(abs_critic_grad[1]).item()},j)
            writer.add_scalars('analysis/critic_grad',{'conv3': torch.mean(abs_critic_grad[2]).item()},j)
            writer.add_scalars('analysis/critic_grad',{'f': torch.mean(abs_critic_grad[3]).item()},j)

            writer.add_scalars('analysis/actor_grad',{'conv1': torch.mean(abs_actor_grad[0]).item()},j)
            writer.add_scalars('analysis/actor_grad',{'conv2': torch.mean(abs_actor_grad[1]).item()},j)
            writer.add_scalars('analysis/actor_grad',{'conv3': torch.mean(abs_actor_grad[2]).item()},j)
            writer.add_scalars('analysis/actor_grad',{'f': torch.mean(abs_actor_grad[3]).item()},j)

            # gradients cancelled out
            conv1_cancel = abs_critic_grad[0] + abs_actor_grad[0] - torch.abs(critic_grad[0] + actor_grad[0])
            conv2_cancel = abs_critic_grad[1] + abs_actor_grad[1] - torch.abs(critic_grad[1] + actor_grad[1])
            conv3_cancel = abs_critic_grad[2] + abs_actor_grad[2] - torch.abs(critic_grad[2] + actor_grad[2])
            f_cancel = abs_critic_grad[3] + abs_actor_grad[3] - torch.abs(critic_grad[3] + actor_grad[3])

            conv1_cancel_mean = torch.mean(conv1_cancel).item()
            conv2_cancel_mean = torch.mean(conv2_cancel).item()
            conv3_cancel_mean = torch.mean(conv3_cancel).item()
            f_cancel_mean = torch.mean(f_cancel).item()

            writer.add_scalars('analysis/cancelled_grad',{'conv1': conv1_cancel_mean},j)
            writer.add_scalars('analysis/cancelled_grad',{'conv2': conv2_cancel_mean},j)
            writer.add_scalars('analysis/cancelled_grad',{'conv3': conv3_cancel_mean},j)
            writer.add_scalars('analysis/cancelled_grad',{'f': f_cancel_mean},j)

            # cancel rate
            writer.add_scalars('analysis/cancelled_rate',{'conv1': conv1_cancel_mean/torch.mean(abs_critic_grad[0]+abs_actor_grad[0]).item()},j)
            writer.add_scalars('analysis/cancelled_rate',{'conv2': conv2_cancel_mean/torch.mean(abs_critic_grad[1]+abs_actor_grad[1]).item()},j)
            writer.add_scalars('analysis/cancelled_rate',{'conv3': conv3_cancel_mean/torch.mean(abs_critic_grad[2]+abs_actor_grad[2]).item()},j)
            writer.add_scalars('analysis/cancelled_rate',{'f': f_cancel_mean/torch.mean(abs_critic_grad[3]+abs_actor_grad[3]).item()},j)


        rollouts.after_update()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == "__main__":
    main()
