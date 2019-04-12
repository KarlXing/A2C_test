import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=32,
                        help='how many training CPU processes to use (default: 32)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=None,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--activation', type=int, default=0,
                        help='activation function for f1 layer, default 0 : relu, 1 : tanh')
    parser.add_argument('--carl-wrapper', action='store_true',default=False,
                        help='use deepmind wrapper or carl wrapper')
    parser.add_argument('--modulation', type=int, default=0,
                        help='whether use modulation, 0: no modulation; 1: input modulation; 2: f1 modulation')
    parser.add_argument('--tanh-f1-tonic', type=float, default=0.5,
                        help='tonic g value for tanh fucntion')
    parser.add_argument('--tanh-f1-phasic', type=float, default=2.0,
                        help='phasic g value for tanh function')
    parser.add_argument('--neuro-input-tonic', type=float, default=100,
                        help='tonic g value for input')
    parser.add_argument('--neuro-input-phasic', type=float, default=20,
                        help='phasic g value for input')
    parser.add_argument('--relu-tonic', type=float, default=0.5,
                        help='tonic g value for relu activation')
    parser.add_argument('--relu-phasic', type=float, default=2.0,
                        help='phasic g value for relu activation')
    parser.add_argument('--log-evaluation', action='store_true', default = False,
                        help='whether log evaluations for later analysis of choosing the threshold')
    parser.add_argument('--phasic-threshold', type=float, default=0.1,
                        help='threshold to switch between phasic and tonic modes')
    parser.add_argument('--input-neuro', action='store_true', default=False,
                        help='whether use norm or neuro activity form of observation')
    parser.add_argument('--sync', action='store_true', default=False,
                        help='only valid in f1 modulation')
    parser.add_argument('--num-eval-processes', type=int, default=10,
                        help='how many processes to use in eval model')
    parser.add_argument('--saved-model', default=None,
                        help='the path of the saved model')
    parser.add_argument('--stats-path', default=None,
                        help='the path to the evaluations results')
    parser.add_argument('--sigmoid-range', type=float, default=5.0,
                        help='the sigmoid curve parameter')
    parser.add_argument('--natural-value', action='store_true', default=False,
                        help='remove abstract of evaluations')
    parser.add_argument('--sigmoid', action='store_true', default=False,
                        help='use sigmoid for g update')
    parser.add_argument('--sleep', type=float, default=0.1,
                        help='sleep time in increasing gpu usage')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
