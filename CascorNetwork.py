from __future__ import absolute_import, division, print_function
from numpy import exp, random, log
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import time
import torch

class CascorNetwork:
    def __init__(self, dataloader, noutputs=1, ninputs=1, max_units=100, unit_type, output_type, use_cache):
        self.max_units = max_units
        self.dataloader = dataloader
        self.unit_type = unit_type
        self.output_type = output_type
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.max_cases = len(self.dataloader.training_inputs)
        self.ncases = self.max_cases
        self.first_case = 0
        self.nunits = 1 + self.ninputs
        self.extra_values = torch.zeros(self.max_units)
        self.extra_values[0] = 1.0
        self.values = self.extra_values
        self.weights = torch.zeros((self.max_units, self.max_units))
        self.outputs = torch.zeros(self.noutputs)
        self.extra_errors = torch.zeros(self.noutputs)
        self.errors = self.extra_errors
        self.sum_errors = torch.zeros(self.noutputs)
        self.dummy_sum_errors = torch.zeros(self.noutputs)
        self.cand_values = torch.zeros(self.ncandidates)
        self.cand_sum_values = torch.zeros(self.ncandidates)
        self.cand_scores = torch.zeros(self.ncandidates)
        if use_cache:
            self.values_cache = torch.zeros((self.max_cases, self.max_units))
            self.errors_cache = torch.zeros((self.max_cases, self.noutputs))

        # For each output, create the vectors holding per-weight info
        self.output_weights = torch.zeros((self.noutputs, self.max_units))
        self.output_weights[:self.noutputs, :1+self.ninputs] = distribution.sample(torch.Size([self.noutputs,
                                                                                               1+self.ninputs]))
        self.output_weights_record = torch.zeros((self.max_units, self.noutputs))
        self.output_deltas = torch.zeros((self.noutputs, self.max_units))
        self.output_slopes = torch.zeros((self.noutputs, self.max_units))
        self.output_prev_slopes = torch.zeros((self.noutputs, self.max_units))

        # For each candidate unit, create the vectors holding the correlations, incoming weights, and other stats
        self.cand_cor = torch.zeros((self.ncandidates, self.noutputs))
        self.cand_prev_cor = torch.zeros((self.ncandidates, self.noutputs))
        self.cand_weights = torch.zeros((self.ncandidates, self.max_units + 1))
        self.cand_deltas = torch.zeros((self.ncandidates, self.max_units + 1))
        self.cand_slopes = torch.zeros((self.ncandidates, self.max_units + 1))
        self.cand_prev_slopes = torch.zeros((self.ncandidates, self.max_units + 1))
        self.cand_derivs = torch.zeros((self.ncandidates, self.max_units + 1))


    def quickprop_update(self, weights, deltas, slopes, prevs, epsilon, decay, mu, shrink_factor, is_input):
        n_columns = self.nunits + 1 if is_input else self.nunits
        n_rows = self.ncandidates if is_input else self.noutputs
        next_step = torch.zeros((n_rows, n_columns))
        w = self.weights[:n_rows, :n_columns]
        d = self.deltas[:n_rows, :n_columns]
        s = self.slopes[:n_rows, :n_columns] + (w * decay)
        p = self.prevs[:n_rows, :n_columns]
        t = self.torch.where(p == s, torch.ones(p.shape), p - s)
        next_step -= torch.where(d * s <= 0, epsilon * s, torch.zeros(next_step.shape))
        mask1 = (((d < 0) & (s >= shrink_factor * p)) | ((d > 0) & (s <= shrink_factor * p))).type(torch.FloatTensor)
        mask2 = (((d < 0) & (s < shrink_factor * p)) | ((d > 0) & (s > shrink_factor * p))).type(torch.FloatTensor)
        next_step += mu * d * mask1
        next_step += (d * s / t) * mask2

        deltas[:n_rows, :n_columns] = next_step
        weights[:n_rows, :n_columns] += next_step
        prevs[:n_rows, :n_columns] = slopes[:n_rows, :n_columns] + (w * decay)
        slopes[:n_rows, :n_columns] *= 0.0


    def set_up_inputs(self, input_vec):
        self.values[0] = 1.0
        self.values[1:self.ninputs+1] = input_vec

    def compute_unit_value(self, j, prev_value):
        w = self.weights[j]
        self.values[j] = self.unit_type.activation_singleton(torch.sum((w[:j] * values[:j])) + prev_value * w[j])
        return self.values[j]

    def output_forward_pass(self):
        self.outputs = torch.matmul(values[:nunits],(output_weights[:, :nunits]).transpose(0,1))
        self.outputs = self.output_type.output_function(outputs)

    def full_forward_pass(self, input_vec, no_memory):
        self.set_up_inputs(input_vec)
        for j in range(1 + self.ninputs, self.nunits):
            if no_memory:
                self.compute_unit_value(self, j, 0.0)
            else:
                self.compute_unit_value(self, j, values[j])
        self.output_forward_pass()



