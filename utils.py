import torch
import torch.nn as nn
import numpy as np

from envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

def tanh_g(x,g):
    x = x/g
    return torch.tanh(x)

def calc_g(evaluation, phasic_g, tonic_g, mean_pos_pderror, mean_neg_pderror):
    if evaluation > 0:
        return min((1 + evaluation/mean_pos_pderror), phasic_g) if mean_pos_pderror > 0 else 1
    elif evaluation < 0:
        return max((1 - evaluation/mean_neg_pderror), tonic_g) if mean_neg_pderror < 0 else 1
    else:
        return 1

def update_mode(evaluations, masks, reward, value, next_value, tonic_g, phasic_g, g, threshold, positive_pderror, negative_pderror):
    value = value.cpu()
    next_value = next_value.cpu()
    pderror = reward-value+next_value
    for i in range(pderror.shape[0]):
        if pderror[i][0] > 0:
            positive_pderror.append(pderror[i][0])
        elif pderror[i][0] < 0:
            negative_pderror.append(pderror[i][0])
    evaluations = 0.75*evaluations + 0.25*pderror
    evaluations = evaluations*masks
    mean_pos_pderror = np.mean(positive_pderror).item() if len(positive_pderror) > 0 else 0
    mean_neg_pderror = np.mean(negative_pderror).item() if len(negative_pderror) > 0 else 0
    for i in range(g.shape[0]):
        g[i][0] = calc_g(evaluations[i], phasic_g, tonic_g, mean_pos_pderror, mean_neg_pderror)
    return evaluations, g, positive_pderror, negative_pderror

def neuro_activity(obs, g, mid = 128):
    assert(obs.shape[0] == g.shape[0])
    for i in range(obs.shape[0]):
        obs[i] = (torch.tanh((obs[i]-mid)/g[i])+1)/2
    return obs

def obs_representation(obs, modulation, g_device, input_neuro):
    if modulation == 0:  # no modulation
        if input_neuro:
            obs = neuro_activity(obs, g_device)
        else:
            obs = obs/255
    elif modulation == 1:  # input modulation
        if input_neuro:
            obs = neuro_activity(obs, g_device)
        else:
            obs = obs/255
            obs = obs/g_device
    else:  # f1 modulation
        obs = obs/255
    return obs
