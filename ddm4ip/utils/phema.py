import copy

import numpy as np
import torch


def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape) # type: ignore
    return exp


def power_function_beta(std, t_next, t_delta):
    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta

#----------------------------------------------------------------------------
# Class for tracking power function EMA during the training.

class PowerFunctionEMA:
    @torch.no_grad()
    def __init__(self, net, stds=[0.050, 0.100]):
        self.net = net
        self.stds = stds
        self.emas = [copy.deepcopy(net) for _std in stds]

    @torch.no_grad()
    def update(self, cur_nimg, batch_size):
        for std, ema in zip(self.stds, self.emas):
            beta = power_function_beta(std=std, t_next=cur_nimg, t_delta=batch_size)
            for p_net, p_ema in zip(self.net.parameters(), ema.parameters()):
                p_ema.lerp_(p_net, 1 - beta)

    @torch.no_grad()
    def get(self):
        for ema in self.emas:
            for p_net, p_ema in zip(self.net.buffers(), ema.buffers()):
                p_ema.copy_(p_net)
        return [(ema, f'-{std:.3f}') for std, ema in zip(self.stds, self.emas)]

    def state_dict(self):
        return dict(stds=self.stds, emas=[ema.state_dict() for ema in self.emas])

    def load_state_dict(self, state):
        self.stds = state['stds']
        for ema, s_ema in zip(self.emas, state['emas']):
            ema.load_state_dict(s_ema)


class FixedBetaEMA:
    """Standard fixed-decay EMA of a network's parameters.

    p_ema <- beta * p_ema + (1 - beta) * p_net, with a constant beta — the
    ubiquitous mainstream EMA (BYOL/MoCo/diffusion), whose horizon is a fixed
    ~1/(1-beta) steps. Simpler to reason about / report than PowerFunctionEMA's
    relative (fraction-of-training) horizon. Holds a deep-copied network so the
    EMA copy can be used like the live net (e.g. ema.get_kernel(...) for val).
    Interface mirrors what the losses need from PowerFunctionEMA.
    """
    @torch.no_grad()
    def __init__(self, net, beta: float = 0.999):
        self.net = net
        self.beta = float(beta)
        self.ema = copy.deepcopy(net)

    @torch.no_grad()
    def update(self):
        for p_net, p_ema in zip(self.net.parameters(), self.ema.parameters()):
            p_ema.lerp_(p_net, 1.0 - self.beta)
        # keep buffers (e.g. non-learnable state) in sync with the live net
        for b_net, b_ema in zip(self.net.buffers(), self.ema.buffers()):
            b_ema.copy_(b_net)

    def state_dict(self):
        return dict(beta=self.beta, ema=self.ema.state_dict())

    def load_state_dict(self, state):
        self.beta = state['beta']
        self.ema.load_state_dict(state['ema'])