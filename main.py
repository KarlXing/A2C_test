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
from utils import get_vec_normalize, neuro_activity, obs_representation, get_g_entropy
from visualize import visdom_plot
from tensorboardX import SummaryWriter

#####################################
# prepare

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

save_path = os.path.join(args.save_dir, args.algo)
try:
    os.makedirs(save_path)
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

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, 4, args.carl_wrapper)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space, args.activation, args.modulation, args.sync,
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
    # print key arguments
    g = (torch.ones(args.num_processes, 1)).to(device)
    #evaluations = torch.zeros(args.num_processes, 1)
    masks_device = torch.ones(args.num_processes, 1).to(device)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size, 1.0)

    obs = envs.reset()
    obs = obs_representation(obs, args.modulation, g_device, args.input_neuro)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    #entropys = deque(maxlen=100)
    #mean_entropy = torch.tensor(0.0)
    start = time.time()
    g_step = 0
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            g_step += 1
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, dist_entropy = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                        args.action_selection)
            obs, reward, done, infos = envs.step(action)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

            obs = obs_representation(obs, args.modulation, g_device, args.input_neuro)

            g = get_g_entropy(dist_entropy)

            if args.log_evaluation:
                writer.add_scalar('analysis/reward', reward[0], g_step)
                writer.add_scalar('analysis/entropy', dist_entropy[0], g_step)
                writer.add_scalar('analysis/maxg', torch.max(g).item(), g_step)
                writer.add_scalar('analysis/ming', torch.min(g).item(), g_step)
                writer.add_scalar('analysis/meang', torch.mean(g).item(), g_step)
                if done[0]:
                    writer.add_scalar('analysis/done', 1, g_step)

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
                        torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))                        


            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, g)

        with torch.no_grad():
            masks_device.copy_(masks)
            next_value = actor_critic.get_value(obs, recurrent_hidden_states, masks_device).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()


    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    main()
