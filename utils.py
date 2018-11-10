import torch
import torch.nn as nn

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

def update_mode(evaluations, masks, reward, value, next_value, tonic_g, phasic_g, g, threshold):
    value = value.cpu()
    next_value = next_value.cpu()
    evaluations = 0.75*evaluations + 0.25*(reward-value+next_value)
    evaluations = evaluations*masks
    for i in range(g.shape[0]):
        g[i][0] = phasic_g if abs(evaluations[i][0]) > threshold else tonic_g
    return evaluations, g

def neuro_activity(obs, g, mid = 128):
    assert(obs.shape[0] == g.shape[0])
    for i in range(obs.shape[0]):
        obs[i] = (torch.tanh((obs[i]-mid)/g[i])+1)/2
    return obs
