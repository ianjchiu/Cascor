from __future__ import absolute_import, division, print_function
from CascorUtil import quickprop_update, compute_unit_value
from HiddenUnit import HiddenUnit, AsigmoidHiddenUnit, GaussianHiddenUnit, SigmoidHiddenUnit
from OutputUnit import LinearOutputUnit, SigmoidOutputUnit
from numpy import exp, random, log
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import time
import torch

class CascorNetwork:
    def __init__(self, use_cache, distribution, dataloader, noutputs=1, ninputs=1, max_units=100, unit_type, output_type, use_cache):
        self.use_cache = use_cache
        self.distribution = distribution
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
        if self.use_cache:
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

    def install_new_unit(self, unit, prev_cor, weight_multiplier):
        if self.nunits >= self.max_units:
            return False
        self.weights[self.nunits, :self.nunits+1] = unit[:1+nunits]
        print("  Add unit {0}: {1}".format(nunits - ninputs, unit))
        self.output_weights[:, self.nunits] = weight_multiplier *  -prev_cor
        if self.use_cache:
            prev_value = 0.0
            for i in range(self.max_cases):
                self.values = self.valeus_cache[i]
                if self.dataloader.use_training_breaks and self.dataloader.training_breaks[i]:
                    prev_value = compute_unit_value(self.values, self.weights, self.unit_type, self.nunits, 0.0)
                else:
                    prev_value = compute_unit_value(self.values, self.weights, self.unit_type, self.nunits, prev_value)
        self.nunits += 1
        return True


    def set_up_inputs(self, input_vec):
        self.values[0] = 1.0
        self.values[1:self.ninputs+1] = input_vec


    def compute_unit_value(self, j, prev_value):
        w = self.weights[j]
        self.values[j] = self.unit_type.activation_singleton(torch.sum((w[:j] * self.values[:j])) + prev_value * w[j])
        return self.values[j]


    def output_forward_pass(self):
        self.outputs = torch.matmul(self.values[:nunits],(self.output_weights[:, :self.nunits]).transpose(0,1))
        self.outputs = self.output_type.output_function(self.outputs)

    def recompute_errors(self, goal):
        for j in range(self.noutputs):
            out = self.outputs[j]
            diff = out - goal[j]
            err_prime = dif * self.output_type.output_prime(out)
            self.errors[j] = err_prime


    def full_forward_pass(self, input_vec, no_memory):
        self.set_up_inputs(input_vec)
        for j in range(1 + self.ninputs, self.nunits):
            if no_memory:
                self.compute_unit_value(self.values, self.weights, self.unit_type, j, 0.0)
            else:
                self.compute_unit_value(self.values, self.weights, self.unit_type,j, values[j])
        self.output_forward_pass()




