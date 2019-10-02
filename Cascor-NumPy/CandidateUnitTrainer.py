from __future__ import absolute_import, division, print_function
import numpy as np
from CascorUtil import CascorUtil


class CandidateUnitTrainer:
    """Trains the candidate unit pool"""

    def __init__(self, network, input_patience, input_change_threshold, input_shrink_factor,
                 input_mu, input_decay, input_epsilon, stats):
        """Give new random weights to all of the candidate units.  Zero the other
          candidate-unit statistics."""
        self.stats = stats
        self.decay = input_decay
        self.epsilon = input_epsilon
        self.shrink_factor = input_shrink_factor
        self.mu = input_mu
        self.network = network
        self.patience = input_patience
        self.change_threshold = input_change_threshold
        self.best_candidate = 0
        self.best_candidate_score = 0.0
        self.network.errors = np.zeros(self.network.noutputs)
        self.cand_values = np.zeros(self.network.ncandidates)
        self.cand_sum_values = np.zeros(self.network.ncandidates)
        self.cand_scores = np.zeros(self.network.ncandidates)
        self.cand_deltas = np.zeros((self.network.ncandidates, self.network.nunits + 1))
        self.cand_derivs = np.zeros((self.network.ncandidates, self.network.nunits + 1))
        self.cand_slopes = np.zeros((self.network.ncandidates, self.network.nunits + 1))
        self.cand_prev_slopes = np.zeros((self.network.ncandidates, self.network.nunits + 1))
        self.cand_cor = np.zeros((self.network.ncandidates, self.network.noutputs))
        self.cand_prev_cor = np.zeros((self.network.ncandidates, self.network.noutputs))
        self.cand_weights = self.network.distribution(-1,1,np.Size([self.network.ncandidates, self.network.nunits + 1]))

    def reset_candidates(self):
        """Give new random weights to all of the candidate units.  Zero the other
          candidate-unit statistics."""
        self.cand_values = np.zeros(self.network.ncandidates)
        self.cand_sum_values = np.zeros(self.network.ncandidates)
        self.cand_scores = np.zeros(self.network.ncandidates)
        self.cand_deltas = np.zeros((self.network.ncandidates, self.network.nunits + 1))
        self.cand_derivs = np.zeros((self.network.ncandidates, self.network.nunits + 1))
        self.cand_slopes = np.zeros((self.network.ncandidates, self.network.nunits + 1))
        self.cand_prev_slopes = np.zeros((self.network.ncandidates, self.network.nunits + 1))
        self.cand_cor = np.zeros((self.network.ncandidates, self.network.noutputs))
        self.cand_prev_cor = np.zeros((self.network.ncandidates, self.network.noutputs))
        self.cand_weights = \
            self.network.distribution(self.network.ncandidates, self.network.nunits + 1) * 2 * \
            self.network.weight_range - self.network.weight_range

    def compute_correlations(self, no_memory):
        """For the current training pattern, compute the value of each candidate
          unit and begin to compute the correlation between that unit's value and
          the error at each output.  We have already done a forward-prop and
          computed the error values for active units."""
        acc_sum = np.matmul(self.cand_weights[:, 0:self.network.nunits],
                               np.reshape(self.network.values[:self.network.nunits], (self.network.nunits,1)))
        if not no_memory:
            acc_sum += np.reshape(self.cand_weights[:, self.network.nunits],
                                  (self.cand_weights[:, self.network.nunits].shape[0], 1)) *  \
                       np.reshape(self.cand_values,(self.cand_values.shape[0], 1))
        v = self.network.unit_type.activation(acc_sum)
        vt = np.reshape(v, v.shape[0])
        self.cand_values[:] = vt
        self.cand_sum_values[:] += vt
        self.cand_cor[:] += v * self.network.errors

    def adjust_correlations(self):
        """Normalize each accumulated correlation value, and stuff the normalized
      form into the cand_prev_cor data structure.  Then zero cand_cor to
      prepare for the next round.  Note the unit with the best total
      correlation score."""
        self.best_candidate = 0
        self.best_candidate_score = 0.0
        if self.network.sum_sq_error == 0.0:
            self.best_candidate_score = max(self.best_candidate_score, 0.0)
        else:
            self.cand_prev_cor[:self.network.ncandidates, :self.network.noutputs] = \
                (self.cand_cor - self.network.sum_errors * self.cand_sum_values.unsqueeze(1)) / \
                self.network.sum_sq_error
            self.cand_scores[:self.network.ncandidates] = (np.abs(self.cand_prev_cor)).sum(1)
            self.cand_cor = np.zeros((self.network.ncandidates, self.network.noutputs))
            (cur_best, index) = np.max(self.cand_scores, 0)
            if cur_best >= self.best_candidate_score:
                self.best_candidate_score = cur_best.item()
                self.best_candidate = index.item()

    def compute_slopes(self, no_memory):
        """Given the correlation values for each candidate-output pair, compute
      the derivative of the candidate's score with respect to each incoming
      weight."""
        acc_sum = np.matmul(self.cand_weights[:self.network.ncandidates, :self.network.nunits],
                               (self.network.values[:self.network.nunits]).reshape((self.network.nunits, 1)))
        acc_sum = acc_sum.view(acc_sum.shape[0])
        if not no_memory:
            acc_sum += self.cand_weights[:self.network.ncandidates, self.network.nunits] * \
                       self.cand_values[:self.network.ncandidates]
        value = self.network.unit_type.activation(acc_sum)
        actprime = self.network.unit_type.activation_prime(value, acc_sum)
        # Now compute which way we want to adjust each unit's incoming activation sum and how much
        if self.network.sum_sq_error == 0:
            direction = np.zeros(self.network.ncandidates)
        else:
            direction = (np.where(self.cand_prev_cor == 0, np.zeros((self.network.ncandidates,
                                                                           self.network.noutputs)),
                                     -1 * np.sign(self.cand_prev_cor) * ((self.network.errors -
                                                                             self.network.sum_errors) /
                                                                            self.network.sum_sq_error).repeat(
                                         self.network.ncandidates, 1))).sum(1)
            direction = direction.view(direction.shape[0])
        self.cand_cor += self.network.errors * value.view((value.shape[0], 1))
        if no_memory:
            dsum = actprime.view(actprime.shape[0], 1) * self.network.values[:self.network.nunits]
        else:
            dsum = actprime.view(actprime.shape[0], 1) * \
                   (self.network.values[:self.network.nunits] + (self.cand_weights[:, self.network.nunits] *
                                            self.cand_derivs[:, :self.network.nunits].transpose(0, 1)).transpose(0, 1))
        self.cand_slopes[:self.network.ncandidates, :self.network.nunits] += \
            direction.view((direction.shape[0], 1)) * dsum
        self.cand_derivs[:self.network.ncandidates, :self.network.nunits] = dsum

        if not no_memory:
            dsum = actprime * (self.cand_values + self.cand_weights[:self.network.ncandidates, self.network.nunits] *
                               self.cand_derivs[:self.network.ncandidates, self.network.nunits])
            self.cand_slopes[:self.network.ncandidates, self.network.nunits] += direction * dsum
            self.cand_derivs[:self.network.ncandidates, self.network.nunits] = dsum
            # Compute derivative of activation sum w.r.t unit's auto-recurrent weight
        # Save unit value for use in next training case
        self.cand_values = value
        self.cand_sum_values += value

    def update_input_weights(self):
        """Update the input weights, using the pre-computed slopes, prev_slopes,
      and delta values.  Uses the quickprop update function."""
        eps = self.epsilon / (self.network.ncases * self.network.nunits)
        CascorUtil.quickprop_update(self.cand_weights, self.network.nunits, self.network.ncandidates,
                                    self.network.noutputs, self.cand_deltas, self.cand_slopes,
                                    self.cand_prev_slopes, eps, self.decay,
                                    self.mu, self.shrink_factor, True)

    def train_inputs_epoch(self):
        """For each training pattern, perform a forward pass. Tune the candidate units'
          weights to maximize the correlation score of each"""
        self.cand_values = np.zeros(self.network.ncandidates)
        self.cand_sum_values = np.zeros(self.network.ncandidates)
        if not self.network.use_cache:
            self.network.values = self.network.extra_values
            self.network.errors = self.network.extra_errors
            self.network.values[1 + self.network.ninputs:self.network.nunits] = \
                np.zeros(self.network.nunits - 1 - self.network.ninputs)
        # Now run through all the training examples
        for i in range(self.network.first_case, self.network.first_case + self.network.ncases):
            # Compute values and errors, or recall cached values.
            no_memory = self.network.dataloader.use_training_breaks and self.network.dataloader.training_breaks[i]
            if self.network.use_cache:
                self.network.values = self.network.values_cache[i]
                self.network.errors = self.network.errors_cache[i]
            else:
                self.network.full_forward_pass(self.network.dataloader.training_inputs[i], no_memory)
                self.network.recompute_errors(self.network.dataloader.training_outputs[i])
            # Compute the slopes we will use to adjust candidate weights.
            self.compute_slopes(no_memory)
        self.shrink_factor = self.mu / (1.0 + self.mu)
        # Now adjust the candidate unit input weights using quickprop
        self.update_input_weights()
        # Fix up the correlation values for the next epoch
        self.adjust_correlations()
        self.stats.epoch += 1

    def correlations_epoch(self):
        """Do an epoch through all active training patterns just to compute the
         initial correlations.  After this one pass, we will update the
         correlations as we train."""
        self.cand_values[:self.network.ncandidates] = 0.0
        self.cand_sum_values[:self.network.ncandidates] = 0.0
        if not self.network.use_cache:
            self.network.values = self.network.extra_values
            self.network.errors = self.network.extra_errors
            self.network.values[(1 + self.network.ninputs):self.network.nunits] = 0.0

        for i in range(self.network.first_case, self.network.first_case + self.network.ncases):
            no_memory = self.network.dataloader.use_training_breaks and self.network.dataloader.training_breaks[i]
            if self.network.use_cache:
                self.network.values = self.network.values_cache[i]
                self.network.errors = self.network.errors_cache[i]
            else:
                self.network.full_forward_pass(self.network.dataloader.training_inputs[i], no_memory)
                self.network.recompute_errors(self.network.dataloader.training_outputs[i])
            self.compute_correlations(no_memory)
        self.adjust_correlations()
        self.stats.epoch += 1


    def train_inputs(self, max_epochs):
        """Train the input weights of all candidates.  If we exhaust max_epochs,
          stop with value "timeout".  Else, keep going until the best candidate
          unit's score has changed by a significant amount, and then until it does
          not change significantly for patience epochs.  Then return stagnant.  If
          patience is zero, we do not stop until victory or until max_epochs is
          used up."""
        self.network.sum_errors /= self.network.ncases
        self.correlations_epoch()
        last_score = 0.0
        stop = max_epochs
        first_time = True
        for i in range(max_epochs):
            self.train_inputs_epoch()
            if self.patience == 0:
                continue
            elif first_time:
                first_time = False
                last_score = self.best_candidate_score
            elif abs(self.best_candidate_score - last_score) > (last_score * self.change_threshold):
                last_score = self.best_candidate_score
                stop = i + self.patience
            elif i >= stop:
                # stagnant
                return "stagnant"
        # timeout
        return "timeout"

