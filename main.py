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
from utils import get_vec_normalize, update_mode, neuro_activity, obs_representation
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
    best_eval = 0

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
    print("modulation type ", args.modulation)
    if not args.input_neuro:
        print("use normalized input")
    else:
        print("use neuro activity of input")
    if args.modulation == 0:
        tonic_g = args.neuro_input_tonic
        phasic_g = args.neuro_input_phasic
    elif args.modulation == 1:
        if args.input_neuro:
            tonic_g = args.neuro_input_tonic
            phasic_g = args.neuro_input_phasic
        else:
            tonic_g = args.relu_tonic
            phasic_g = args.relu_phasic
    elif args.modulation == 2:
        if args.activation == 0:
            tonic_g = args.relu_tonic
            phasic_g = args.relu_phasic
        else:
            tonic_g = args.tanh_f1_tonic
            phasic_g = args.tanh_f1_phasic
    else:
        print("invalid modulation")

    g = torch.ones(args.num_processes, 1)*tonic_g
    g_device = (torch.ones(args.num_processes, 1)*tonic_g).to(device)
    evaluations = torch.zeros(args.num_processes, 1)
    masks_device = torch.ones(args.num_processes, 1).to(device)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size, tonic_g)

    obs = envs.reset()
    obs = obs_representation(obs, args.modulation, g_device, args.input_neuro)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.g[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

            obs = obs_representation(obs, args.modulation, g_device, args.input_neuro, )

            #update g
            with torch.no_grad():
                masks_device.copy_(masks)
                next_value = actor_critic.get_value(obs, g_device, recurrent_hidden_states, masks_device).detach()
            evaluations, g = update_mode(evaluations, masks, reward, value, next_value, tonic_g, phasic_g, g, args.phasic_threshold)
            if args.modulation != 0:
                g_device.copy_(g)

            if args.log_evaluation:
                writer.add_scalar('analysis/evaluations', evaluations[0], j*args.num_steps + step)
            for idx in range(len(infos)):
                info = infos[idx]
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    steps_done = j*args.num_steps*args.num_processes + step*args.num_processes + idx
                    writer.add_scalar('data/reward', info['episode']['r'], steps_done )

            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, g_device)

        # with torch.no_grad():
        #     next_value = actor_critic.get_value(rollouts.obs[-1],
        #                                         rollouts.recurrent_hidden_states[-1],
        #                                         rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # if j % args.save_interval == 0 and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass

        #     # A really ugly way to save a model to CPU
        #     save_model = actor_critic
        #     if args.cuda:
        #         save_model = copy.deepcopy(actor_critic).cpu()

        #     save_model = [save_model,
        #                   getattr(get_vec_normalize(envs), 'ob_rms', None)]

        #     torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        # total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if args.log_interval is not None and j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))

        if (args.eval_interval is not None
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_eval_processes, args.num_eval_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True, 4, args.carl_wrapper)

            # MsPacman no vecnormlize
            vec_norm = get_vec_normalize(eval_envs)
            assert(vec_norm is None)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            eval_recurrent_hidden_states = torch.zeros(args.num_eval_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks_device = torch.zeros(args.num_eval_processes, 1, device=device)
            eval_g = torch.ones(args.num_eval_processes, 1)*tonic_g
            eval_g_device = torch.ones(args.num_eval_processes, 1, device=device)*tonic_g
            eval_evaluations = torch.zeros(args.num_eval_processes, 1)
            obs = eval_envs.reset()
            obs = obs_representation(obs, args.modulation, eval_g_device, args.input_neuro)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    value, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_g_device, eval_recurrent_hidden_states, eval_masks_device, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                obs = obs_representation(obs, args.modulation, eval_g_device, args.input_neuro)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                #update eval_g
                with torch.no_grad():
                    eval_masks_device.copy_(eval_masks)
                    next_value = actor_critic.get_value(obs, eval_g_device, eval_recurrent_hidden_states, eval_masks_device).detach()
                eval_evaluations, eval_g = update_mode(eval_evaluations, eval_masks, reward, value, next_value, tonic_g, phasic_g, eval_g, args.phasic_threshold)
                if args.modulation != 0:
                    eval_g_device.copy_(eval_g)

                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()
            # print("eval scores are ", eval_episode_rewards)
            mean_eval = np.mean(eval_episode_rewards)
            if mean_eval > best_eval:
                best_eval = mean_eval
                save_model = actor_critic
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()
                torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
            writer.add_scalar('data/eval_reward', mean_eval, j)
            # print(" Evaluation using {} episodes: mean reward {:.5f}\n".
            #     format(len(eval_episode_rewards),
            #            np.mean(eval_episode_rewards)))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass
    print("best eval score is ", best_eval)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    main()
