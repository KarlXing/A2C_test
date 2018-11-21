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
torch.manual_seed(args.seed)
device = "cpu"
try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

######################################
# main

def main():
    writer = SummaryWriter()
    torch.set_num_threads(1)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, 4, args.carl_wrapper)

    assert(args.saved_model is not None)
    actor_critic = torch.load(args.saved_model)

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
    print("tonic g is: ", tonic_g)
    print("phasic g is: ", phasic_g)

    g = torch.ones(args.num_processes, 1)*tonic_g
    evaluations = torch.zeros(args.num_processes, 1)
    masks = torch.ones(args.num_processes, 1)
    recurrent_hidden_states = torch.zeros(args.num_processes, actor_critic.recurrent_hidden_state_size)

    all_evaluations = []

    obs = envs.reset()
    obs = obs_representation(obs, args.modulation, g, args.input_neuro)

    eval_episode_rewards = []

    while(len(eval_episode_rewards) < 10):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        obs, g, recurrent_hidden_states, masks ,deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

        obs = obs_representation(obs, args.modulation, g, args.input_neuro)

        #update g
        with torch.no_grad():
            next_value = actor_critic.get_value(obs, g, recurrent_hidden_states, masks).detach()
        evaluations, g = update_mode(evaluations, masks, reward, value, next_value, tonic_g, phasic_g, g, args.phasic_threshold)
        all_evaluations.append(evaluations)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
    print(eval_episode_rewards)
    assert(args.stats_path is not None)
    torch.save([all_evaluations, eval_episode_rewards], args.stats_path)

if __name__ == "__main__":
    main()
