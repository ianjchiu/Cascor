from __future__ import absolute_import, division, print_function
import numpy as np
import torch

class OutputUnit:
    def __init__(self):
        pass

    def output_function(self, acc_sum):
        """Compute the value of an output, given the weighted sum of incoming values."""
        pass


    def output_prime(self, output):
        """Compute the derivative of an output with respect to the weighted sum of incoming values."""
        pass


class SigmoidOutputUnit(OutputUnit):
    def __init__(self):
        pass


    def output_function(self, acc_sum):
        with np.errstate(divide='ignore', over='ignore', under='ignore'):
            return torch.where(torch.abs(acc_sum) <= 15, (1.0 / (1.0 + torch.exp(-acc_sum))) - 0.5,
                               torch.sign(acc_sum) * 0.5)


    def output_prime(self, output):
        return sigmoid_prime_offset + 0.25 - output * output


class LinearOutputUnit(OutputUnit):
    def __init__(self):
        pass


    def output_function(self, acc_sum):
        return acc_sum


    def output_prime(self, output):
        return 1.0