from __future__ import absolute_import, division, print_function
import numpy as np
import torch


class HiddenUnit:
    def __init__(self):
        pass

    def activation_singleton(self, acc_sum):
        """Given the sum of weighted inputs, compute the unit's activation value."""
        pass

    def activation(self, acc_sum):
        """Given a vector of the units' sum of weighted inputs, compute the unit's activation value."""
        pass

    def activation_prime(self, value, acc_sum):
        """Given a vector of units' activation value and sum of weighted inputs, compute
         the derivative of the activation with respect to the sum. """
        pass


class SigmoidHiddenUnit(HiddenUnit):
    def __init__(self):
        pass

    def activation_singleton(self, acc_sum):
        if acc_sum < -15.0:
            return -0.5
        elif acc_sum > 15:
            return 0.5
        else:
            return (1.0 / (1.0 + np.exp(-acc_sum))) - 0.5

    def activation(self, acc_sum):
        with np.errstate(divide='ignore', over='ignore', under='ignore'):
            return torch.where((torch.abs(acc_sum) <= 15), (1.0 / (1.0 + torch.exp(-acc_sum))) - 0.5, torch.sign(
                acc_sum) * 0.5)

    def activation_prime(self, value, acc_sum):
        return .25 - value * value


class AsigmoidHiddenUnit(HiddenUnit):
    def activation_singleton(self, acc_sum):
        if acc_sum < -15.0:
            return 0.0
        elif acc_sum > 15.0:
            return 1.0
        else:
            return 1.0 / (1.0 + np.exp(-acc_sum))

    def activation(self, acc_sum):
        with np.errstate(divide='ignore', over='ignore', under='ignore'):
            return torch.where((torch.abs(acc_sum) <= 15), (1.0 / (1.0 + torch.exp(-acc_sum))), \
                               torch.sign(acc_sum) * 0.5 + 0.5)

    def activation_prime(self, value, acc_sum):
        return value * (1.0 - value)


class GaussianHiddenUnit(HiddenUnit):
    def __init__(self):
        pass

    def activation_singleton(self, acc_sum):
        tmp = -0.5 * acc_sum * acc_sum
        if tmp < -75.0:
            return 0.0
        else:
            return np.exp(tmp)

    def activation(self, acc_sum):
        with np.errstate(divide='ignore', over='ignore', under='ignore'):
            tmp = -0.5 * acc_sum * acc_sum
            return torch.where((tmp < -75.0), torch.zeros(tmp.shape()), torch.exp(tmp))

    def activation_prime(self, value, acc_sum):
        return acc_sum * (-value)