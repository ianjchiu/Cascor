from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from HiddenUnit import HiddenUnit, GaussianHiddenUnit, AsigmoidHiddenUnit, SigmoidHiddenUnit

def quickprop_update(weights, nunits, ncandidates, noutputs, deltas, slopes, prevs, epsilon, decay, mu, shrink_factor, is_input):
    n_columns = nunits + 1 if is_input else nunits
    n_rows = ncandidates if is_input else noutputs
    next_step = torch.zeros((n_rows, n_columns))
    w = weights[:n_rows, :n_columns]
    d = deltas[:n_rows, :n_columns]
    s = slopes[:n_rows, :n_columns] + (w * decay)
    p = prevs[:n_rows, :n_columns]
    t = torch.where(p == s, torch.ones(p.shape), p - s)
    next_step -= torch.where(d * s <= 0, epsilon * s, torch.zeros(next_step.shape))
    mask1 = (((d < 0) & (s >= shrink_factor * p)) | ((d > 0) & (s <= shrink_factor * p))).type(torch.FloatTensor)
    mask2 = (((d < 0) & (s < shrink_factor * p)) | ((d > 0) & (s > shrink_factor * p))).type(torch.FloatTensor)
    next_step += mu * d * mask1
    next_step += (d * s / t) * mask2

    deltas[:n_rows, :n_columns] = next_step
    weights[:n_rows, :n_columns] += next_step
    prevs[:n_rows, :n_columns] = slopes[:n_rows, :n_columns] + (w * decay)
    slopes[:n_rows, :n_columns] *= 0.0

def compute_unit_value(values, weights, unit_type, j, prev_value):
    w = weights[j]
    values[j] = unit_type.activation_singleton(torch.sum((w[:j] * values[:j])) + prev_value * w[j])
    return values[j]