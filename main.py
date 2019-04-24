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
from utils import get_vec_normalize, update_mode_entropy, neuro_activity
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
    # print key arguments
    g_device = (torch.ones(args.num_processes, 1)).to(device)
    masks_device = torch.ones(args.num_processes, 1).to(device)
    reward_count = 0
    start_step = 0

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size, 1.0)

    obs = envs.reset()
    obs = obs/255
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    start = time.time()
    g_step = 0
    reward_history = set()
    primitive_reward_history = set()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            g_step += 1
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, xmin, xmax, xmean, ori_dist_entropy, ratio = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
            ori_dist_entropy = ori_dist_entropy.cpu().unsqueeze(1)
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)


            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

            obs = obs/255

            if reward[0] != 0:
                reward_count += 1
            # the log info is based on subprocess 0
            if args.log_evaluation:
                # writer.add_scalar('analysis/ratio', ratio/args.num_processes, g_step)
                # writer.add_scalar('analysis/reward', reward[0], g_step)
                writer.add_scalar('analysis/entropy', ori_dist_entropy[0].item(), g_step)
                # writer.add_scalar('analysis/xmin', xmin, g_step)
                # writer.add_scalar('analysis/xmax', xmax, g_step)
                # writer.add_scalar('analysis/xmean', xmean, g_step)
                # if done[0]:
                #     writer.add_scalar('analysis/done', 1, g_step)
                # if 'episode' in infos[0].keys():
                #     writer.add_scalar('analysis/reward_density', reward_count/(g_step - start_step), g_step)
                #     reward_count = 0
                #     start_step = g_step + 1
                for info in infos:
                    if 'new_reward' in info:
                        new_rewards  = info['new_reward'] - primitive_reward_history
                        if len(new_rewards) > 0:
                            print('new primitive rewards: ', new_rewards, ' time: ', g_step)
                            primitive_reward_history =  primitive_reward_history.union(info['new_reward'])

            if args.track_reward:
                for r in reward:
                    r = r.item()
                    if r not in reward_history:
                        print('new step rewards: ', r, g_step)
                        reward_history.add(r)


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
            assert(torch.sum(g_device).item() == args.num_processes)
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, g_device)

        with torch.no_grad():
            masks_device.copy_(masks)
            next_value, next_dist_entropy = actor_critic.get_value(obs, recurrent_hidden_states, masks_device)

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    main()
