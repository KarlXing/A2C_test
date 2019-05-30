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
from utils import update_base_reward, RunningMeanStd

#####################################
# prepare

now = datetime.datetime.now().isoformat()

args = get_args()

assert args.algo == 'ppo'

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

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, 4, True, args.track_primitive_reward)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space, args.activation,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)


    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps, max_grad_norm=args.max_grad_norm)


    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    rwd_rms = RunningMeanStd()
    rwd_rms.to(device)
    obs_rms = RunningMeanStd(shape=(1,1,84,84))
    obs_rms.to(device)

    # initiate env and storage rollout
    obs = envs.reset()
    obs = obs/255
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # init rms 
    assert((rollouts.obs.size()[2] == 4) and len(rollouts.obs.size()) == 5)
    for step in range(args.num_steps * args.pre_obs_rms_steps):
        actions = torch.from_numpy(np.random.randint(0, envs.action_space.n, size=(args.num_processes,))).unsqueeze(1)
        obs, reward, done, infos  = envs.step(actions)

        rollouts.insert_obs(obs)
        if step % args.num_steps == 0:
            obs_rms.update(rollouts.obs[:,:,3:,:,:].view(-1, 1, 84, 84))  

    # clean rollouts before start to train
    rollouts.clean()

    # necessary variabels
    episode_rewards = deque(maxlen=10) # store last 10 episode rewards
    g_step = 0 # global step
    masks_device = torch.ones(args.num_processes, 1).to(device)  # mask on gpu
    intrinsic_reward = 0.0

    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            g_step += 1
            with torch.no_grad():
                value_ex, value_in, action, action_log_prob, recurrent_hidden_states, entropy, f_a = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward_ex, done, infos = envs.step(action)
            obs = obs/255
            rnd_obs = obs[:,3:,:,:] # only the last frame

            with torch.no_grad():
                reward_in = actor_critic.compute_intrinsic_reward(((rnd_obs - obs_rms.mean) / torch.sqrt(obs_rms.var)).clamp(-5, 5)).unsqueeze(1)
            # reward_in = reward_in / torch.sqrt(rwd_rms.var)
            # rescale intrinsic rewards after all the steps

            intrinsic_reward += reward_in[0].item()
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

            if args.log_evaluation:
                writer.add_scalar('analysis/entropy', entropy.mean().item(), g_step)
                writer.add_scalar('analysis/reward_in', reward_in[0].item(), g_step)
                writer.add_scalar('analysis/action_log_prob', torch.exp(action_log_prob[0].item()), g_step)

            for idx in range(len(infos)):
                info = infos[idx]
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    steps_done = g_step*args.num_processes + idx
                    writer.add_scalar('data/reward', info['episode']['r'], steps_done)
                    mean_rewards = np.mean(episode_rewards)
                    writer.add_scalar('data/avg_reward', mean_rewards, steps_done)
                    if idx == 0:
                        writer.add_scalar('analysis/reward_in_episode', intrinsic_reward, steps_done)
                        intrinsic_reward = 0.0
                    if mean_rewards > best_score:
                        best_score = mean_rewards
                        save_model = actor_critic
                        if args.cuda:
                            save_model = copy.deepcopy(actor_critic).cpu()
                        torch.save(save_model, os.path.join(save_path, args.env_name + now + ".pt"))
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value_ex, value_in, reward_ex, reward_in, masks)

        # update obs_rms
        rnd_obs_batch = rollouts.obs[1:].view(-1, *envs.observation_space.shape)[:,3:,:,:]
        rnd_obs_mean = torch.mean(rnd_obs_batch, dim=0)
        rnd_obs_std = torch.std(rnd_obs_batch, dim=0)
        rnd_obs_cnt = rnd_obs_batch.size()[0]
        obs_rms.update_from_moments(rnd_obs_mean, rnd_obs_std, rnd_obs_cnt)

        # update rwd_rms and update reward_in
        reward_in_batch = rollouts.rewards_in.view(-1, 1)
        reward_in_mean = torch.mean(reward_in_batch, dim=0)
        reward_in_std = torch.std(reward_in_batch, dim=0)
        reward_in_cnt = reward_in_batch.size()[0]
        rwd_rms.update_from_moments(reward_in_mean, reward_in_std, reward_in_cnt)

        rollouts.rewards_in = rollouts.rewards_in / torch.sqrt(rwd_rms.var)

        with torch.no_grad():
            masks_device.copy_(masks)
            value_ex, value_in = actor_critic.get_value(obs, recurrent_hidden_states, masks_device)

        rollouts.compute_returns(value_ex, value_in, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy, rnd_loss = agent.update(rollouts, obs_rms)

        rollouts.after_update()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == "__main__":
    main()
