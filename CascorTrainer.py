from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from Dataloader import Dataloader
from OutputUnit import OutputUnit, SigmoidOutputUnit, LinearOutputUnit
from HiddenUnit import HiddenUnit, GaussianHiddenUnit, AsigmoidHiddenUnit, SigmoidHiddenUnit
from datetime import datetime, date
import time
from CascorUtil import CascorUtil, CascorStats
from CascorNetwork import CascorNetwork
from CandidateUnitTrainer import CandidateUnitTrainer


class CascorTrainer:
    def __init__(self, network, candidate_trainer, outlimit, inlimit, rounds, output_patience, output_epsilon,
                 output_mu, output_decay, output_deltas, output_slopes, output_prev_slopes, output_shrink_factor,
                 output_change_threshold, stats=CascorStats(), weight_multiplier=1, test_function=None, test=True,
                 restart=False):
        self.output_change_threshold = output_change_threshold
        self.stats = stats
        self.output_patience = output_patience
        self.output_epsilon = output_epsilon
        self.output_mu = output_mu
        self.output_decay = output_decay
        self.output_deltas = output_deltas
        self.output_slopes = output_slopes
        self.output_shrink_factor = output_shrink_factor
        self.output_prev_slopes = output_prev_slopes
        self.weight_multiplier = weight_multiplier
        self.test_function = test_function
        self.restart = restart
        self.outlimit = outlimit
        self.inlimit = inlimit
        self.rounds = rounds
        self.network = network
        self.candidate_trainer = candidate_trainer
        self.error_bits = 0
        self.true_error = 0
        self.sum_sq_error = 0.0
        self.test = test
        self.start_seconds = None
        self.finished_seconds = None

    def compute_errors(self, goal, err_bits, true_err, sum_sq_err, slopes_p):
        """goal is a vector of desired outputs.  Compute and record the error
      statistics, incrementing the err_bits, true_err, and sum_sq_err variables,
      and the proper entry in sum_errors. If slopes_p is true, also compute
      and record the slopes for output weights."""
        for j in range(self.network.noutputs):
            out = self.network.outputs[j]
            dif = out - goal[j]
            err = dif
            if self.network.hyper_error:
                if dif < -.9999999:
                    err = -17.0
                elif dif > .9999999:
                    err = 17.0
                else:
                    err = np.log(((1.0 + dif) / (1.0 - dif)))

            err_prime = err if self.network.raw_error else err * self.network.output_type.output_prime(out)
            if (abs(dif)) >= self.network.score_threshold:
                err_bits += 1
            true_err += dif * dif
            self.network.errors[j] = err_prime
            self.network.sum_errors[j] += err_prime
            sum_sq_err += err_prime * err_prime
            if slopes_p:
                os = self.network.output_slopes[j]
                for i in range(self.network.nunits):
                    os[i] += err_prime * self.network.values[i]
        return err_bits, true_err, sum_sq_err

    def update_output_weights(self):
        """Update the output weights, using the pre-computed slopes, prev-slopes,
      and delta values. Uses the quickprop update function."""
        eps = self.output_epsilon / self.network.ncases
        CascorUtil.quickprop_update(self.network.output_weights, self.network.nunits,
                                    self.network.ncandidates, self.network.noutputs,
                                    self.output_deltas, self.output_slopes,
                                    self.output_prev_slopes, eps, self.output_decay, self.output_mu,
                                    self.output_shrink_factor, False)

    def train_outputs_epoch(self):
        """Perform forward propagation once for each set of weights in the
      training vectors, computing errors and slopes.  Then update the output
      weights"""
        err_bits = 0
        true_err = 0.0
        sum_sq_err = 0.0
        self.network.sum_errors *= 0.0
        self.output_shrink_factor = self.output_mu / (1.0 + self.output_mu)
        if not self.network.use_cache:
            self.network.values = self.network.extra_values
            self.network.errors = self.network.extra_errors
            self.network.values[1 + self.network.ninputs:] *= 0
        # Now run through the training examples
        for i in range(self.network.first_case, self.network.first_case + self.network.ncases):
            if self.network.use_cache:
                self.network.values = self.network.values_cache[i]
                self.network.errors = self.network.errors_cache[i]
                self.network.output_forward_pass()
            else:
                self.network.full_forward_pass(self.network.dataloader.training_inputs[i],
                                               (self.network.dataloader.use_training_breaks and
                                                self.network.dataloader.training_breaks[i]))
            # gotta expand compute_error macro
            err_bits, true_err, sum_sq_err = self.compute_errors(self.network.dataloader.training_outputs[i], err_bits, true_err, sum_sq_err, True)
        self.error_bits = err_bits
        self.true_error = true_err
        self.network.sum_sq_error = sum_sq_err

        # Do not change weights or count epoch if this run was perfect
        if self.error_bits != 0:
            self.update_output_weights()
            self.stats.epoch += 1


    def train_outputs(self, max_epochs):
        """Train the output weights.  If we exhaust max_epochs, stop with value
          "timeout".  If there are zero error bits, stop with value "win".  Else,
          keep going until the true error has not changed by a significant amount
          for output_patience epochs.  Then return "stagnant".  If
          output_patience is zero, we do not stop until victory or until
          max_epochs is used up."""
        last_error = 0.0
        quit_epoch = self.stats.epoch + self.output_patience
        first_time = True
        for _ in range(max_epochs):
            self.train_outputs_epoch()
            if self.error_bits == 0:
                return "win"
            elif self.output_patience == 0:
                continue
            elif first_time:
                first_time = False
                last_error = self.true_error
            elif abs((self.true_error - last_error)) > (last_error * self.output_change_threshold):
                last_error = self.true_error
                quit_epoch = self.stats.epoch + self.output_patience
            elif self.stats.epoch >= quit_epoch:
                return "stagnant"
        return "timeout"

    def train(self):
        if self.restart:
            self.network = CascorNetwork(self.network.distribution, self.network.dataloader,
                                         self.network.noutputs, self.network.ninputs, self.network.max_units,
                                         self.network.unit_type, self.network.output_type, self.network.use_cache)
        if self.network.use_cache:
            self.network.values_cache *= 0
            self.network.values_cache[:, 0] += 1.0
            self.network.values_cache[:, 1:self.network.ninputs + 1] += self.network.dataloader.training_inputs
        start = datetime.now().time()
        self.start_seconds = time.time()
        print(f'Run started at {start}')
        for r in range(self.rounds):
            res = self.train_outputs(self.outlimit)
            if res == "win":
                finished = datetime.now().time()
                finished_seconds = time.time()
                print(
                    f'Victory at {self.stats.epoch} epochs, {self.network.nunits} units, {(self.network.nunits - self.network.ninputs - 1)} hidden, Error {self.true_error}.')
                print(f'Victory achieved at {finished} in '
                      f'{datetime.combine(date.today(), finished) - datetime.combine(date.today(), start)}')
                if self.test:
                    self.test_function()
                return None
            elif res == "timeout":
                print(f'Epoch {self.stats.epoch}: Out Timeout  {self.error_bits} bits wrong, error {self.true_error}.\n')
            elif res == "stagnant":
                print(f'Epoch {self.stats.epoch}: Out Stagnant {self.error_bits} bits wrong, error {self.true_error}.\n')
            else:
                print("Should not be here, invalid res. \n")
                raise RuntimeError("Invalid train_outputs")
            if self.test and "win" == self.test_function():
                return None
            res = self.candidate_trainer.train_inputs(self.inlimit)
            if res == "timeout":
                print(f'Epoch {self.stats.epoch}: In Timeout.  Cor: {self.candidate_trainer.best_candidate_score}')
            elif res == "stagnant":
                print(f'Epoch {self.stats.epoch}: In Stagnant.  Cor: {self.candidate_trainer.best_candidate_score}')
            else:
                print("Should not be here, invalid res. \n")
                raise RuntimeError("Invalid train_inputs")
            if not self.network.install_new_unit(self.candidate_trainer.cand_weights[self.candidate_trainer.best_candidate],
                                                 self.weight_multiplier,
                                                 self.candidate_trainer.cand_prev_cor[self.candidate_trainer.best_candidate]):
                print(f'Could not add more units')
                finished = datetime.now().time()
                finished_seconds = time.time()
                return "lose"
            self.candidate_trainer.reset_candidates()
        finished = datetime.now().time()
        self.finished_seconds = time.time()
        print(f'Lost at {finished} in '
              f'{datetime.combine(date.today(), finished) - datetime.combine(date.today(), start)}')
        return "lose"

    def test_epoch(self, tmp_score_threshold=0.49999):
        """Perform forward propagation once for each set of weights in the training
      and testing vectors.  Reporting the performance.  Do not change any
      weights.  Do not use the caches."""
        tmp = self.network.score_threshold
        self.network.score_threshold = tmp_score_threshold
        self.network.use_cache = False
        self.network.values = self.network.extra_values
        self.network.errors = self.network.extra_errors
        self.network.sum_errors = self.network.dummy_sum_errors
        train_err_bits = 0
        test_err_bits = 0
        train_true_err = 0.0
        test_true_err = 0.0
        sum_sq_err = 0.0
        # Zero context at the start of the training set.
        self.network.values[1 + self.network.ninputs:self.network.nunits] = torch.zeros(self.network.nunits - 1 - self.network.ninputs)

        # Run all training patterns and count errors

        for i in range(len(self.network.dataloader.training_inputs)):
            self.network.full_forward_pass(self.network.dataloader.training_inputs[i],
                                           self.network.dataloader.use_training_breaks and
                                           self.network.dataloader.training_breaks[i])
            train_err_bits, train_true_err, sum_sq_err = \
                self.compute_errors(self.network.dataloader.training_outputs[i], train_err_bits, train_true_err, sum_sq_err, False)
        print(f'Training: {train_err_bits} of {(len(self.network.dataloader.training_inputs))} wrong, error {train_true_err}.')
        # Zero context at the start of the test set.
        self.network.values[1 + self.network.ninputs:self.network.nunits] = torch.zeros(self.network.nunits - 1 - self.network.ninputs)
        if self.network.dataloader.test_inputs is not None:
            for i in range(len(self.network.dataloader.test_inputs)):
                self.network.full_forward_pass(self.network.dataloader.test_inputs[i], self.network.dataloader.use_test_breaks and self.network.dataloader.test_breaks[i])
                test_err_bits, test_true_err, sum_sq_err = \
                    self.compute_errors(self.network.dataloader.test_outputs[i], test_err_bits, test_true_err, sum_sq_err, False)
            print(f'  Test: {test_err_bits} of {(len(self.network.dataloader.test_inputs))} wrong, error {test_true_err}.')
        self.network.score_threshold = tmp

