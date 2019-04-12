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
from utils import get_vec_normalize, neuro_activity, obs_representation
from visualize import visdom_plot
from tensorboardX import SummaryWriter
import math

#####################################
# prepare

args = get_args()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

######################################
# main

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                     args.gamma, args.log_dir, args.add_timestep, device, False, 4, args.carl_wrapper)

    observation_space_shape = (4,210,160)
    action_space = gym.spaces.discrete.Discrete(8)
    actor_critic = Policy(observation_space_shape, action_space, args.activation,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    # agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
    #                            args.entropy_coef, lr=args.lr,
    #                            eps=args.eps, alpha=args.alpha,
    #                            max_grad_norm=args.max_grad_norm)

    # rollouts = RolloutStorage(args.num_steps, args.num_processes,
    #                     observation_space_shape, action_space,
    #                     actor_critic.recurrent_hidden_state_size, 1.0)


    obs = torch.rand(32,4,210,160).to(device)
    obs = obs/255
    recurrent_hidden_states = torch.zeros(args.num_processes, 1)
    masks = torch.ones(args.num_processes, 1)

    while(True):
        time.sleep(args.sleep)
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, xmin, xmax, xmean, ori_dist_entropy, ratio = actor_critic.act(
                        obs,
                        recurrent_hidden_states,
                        masks)


if __name__ == "__main__":
    main()
