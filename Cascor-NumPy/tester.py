from __future__ import absolute_import, division, print_function
from datasets import morse as ds
import numpy as np
from Dataloader import Dataloader
from OutputUnit import SigmoidOutputUnit
from HiddenUnit import SigmoidHiddenUnit
from CascorUtil import CascorStats
from CascorNetwork import CascorNetwork
from CandidateUnitTrainer import CandidateUnitTrainer
from CascorTrainer import CascorTrainer



start_seconds = None
finished_seconds = None
hyper_error = False
raw_error = False
sigmoid_prime_offset = 0.1
weight_range = 1.0
"""Distribution for the randomized weights"""


weight_multiplier = 1.0
"""The output weights for candidate units get an initial value that is the
  negative of the correlation times this factor."""

output_mu = 2.0
"""Mu parameter used for quickprop training of output weights.  The
  step size is limited to mu times the previous step."""

output_shrink_factor = output_mu / (1.0 + output_mu)
"""Derived from output_mu.  Used in computing whether the proposed step is too large."""

output_epsilon = 0.35
"""Controls the amount of linear gradient descent to use in updating
  output weights."""

output_decay = 0.0001
"""This factor times the current weight is added to the slope at the
  start of each output-training epoch.  Keeps weights from growing too big."""

output_patience = 30
"""If we go for this many epochs with no significant change, it's time to
  stop tuning.  If 0, go on forever."""

output_change_threshold = 0.001
"""The error must change by at least this fraction of its old value in
  order to count as a significant change."""

input_mu = 2.0
"""Mu parameter used for quickprop training of input weights.  The
  step size is limited to mu times the previous step."""

input_shrink_factor = input_mu / (1.0 + input_mu)
"""Derived from input_mu; used in computing whether the proposed step is too large"""

input_epsilon = 1.0
"""Controls the amount of linear gradient descent to use in updating
  unit input weights."""

input_decay = 0.0
"""This factor times the current weight is added to the slope at the
  start of each output-training epoch.  Keeps weights from growing too big."""

input_patience = 25
"""If we go for this many epochs with no significant change, it's time to
  stop tuning.  If 0, go on forever."""

input_change_threshold = 0.005
"""The correlation score for the best unit must change by at least
  this fraction of its old value in order to count as a significant
  change."""

# Variables related to error and correlation.

score_threshold = 0.4
"""An output is counted as correct for a given case if the difference
  between that output and the desired value is smaller in magnitude than
  this value."""

error_bits = 0
"""Count number of bits in epoch that are wrong by more than
  score_threshold"""

true_error = 0.0
"""The sum-squared error at the network outputs.  This is the value the
  algorithm is ultimately trying to minimize."""

sum_sq_error = 0.0
"""Accumulate the sum of the squared error values after output
  training phase."""


best_candidate_score = 0.0
"""The best correlation score found among all candidate units being
  trained."""

best_candidate = 0
"""The index of the candidate unit whose correlation score is best
  at present."""



use_cache = True
"""If T, cache the forward-pass values instead of repeatedly
  computing them.  This can save a *lot* of time if all the cached values
  fit into memory."""


epoch = 0
"""Count of the number of times the entire training set has been presented."""



done = False
"""Set to True whenever some problem-specific test function wants to abort processing"""

test = True
"""If true, run a test epoch every so often during output training"""

test_function = None
"""This variable holds a function of no arg that is called at various times
  in the program to test performance.  This test varies from one problem to
  another, so we hang it on a hook."""


single_pass = False
"""When on, pause after next forward/backward cycle."""

single_epoch = False
"""When on, pause after next training epoch"""

step = False
"""Turned briefly to True in order to continue after a pause"""

# The sets of training inputs and outputs are stored in parallel vectors.
# Each element is a vector of short floats, one for each input or output.

training_inputs = ds.training_inputs
training_outputs = ds.training_outputs
use_training_breaks = ds.use_training_breaks
training_breaks = ds.training_breaks
test_inputs = ds.test_inputs
test_outputs = ds.test_outputs
use_test_breaks = ds.use_test_breaks
test_breaks = ds.training_breaks
max_cases = 0
ncases = 0
first_case = 0

max_units = 100
ninputs = ds.ninputs
noutputs = ds.noutputs
nunits = 0
ncandidates = 16
max_cases = len(training_inputs)
ncases = max_cases
first_case = 0
nunits = 1 + ninputs
# array of arrays of floats
# a float array
extra_values = np.zeros(max_units)
values = extra_values
# I'll figure out what to do with weights SoonTM
weights = None
outputs = np.zeros(noutputs)
extra_errors = np.zeros(noutputs)
errors = extra_errors
sum_errors = np.zeros(noutputs)
dummy_sum_errors = np.zeros(noutputs)
cand_values = np.zeros(ncandidates)
cand_sum_values = np.zeros(ncandidates)
cand_scores = np.zeros(ncandidates)

# Only create the cache if use_cache is on; may not have room
if use_cache:
    values_cache = np.zeros((max_cases, max_units))
    errors_cache = np.zeros((max_cases, noutputs))

# For each output, create the vectors holding per-weight info
output_weights = np.zeros((noutputs, max_units))
output_weights_record = np.zeros((max_units, noutputs))
output_deltas = np.zeros((noutputs, max_units))
output_slopes = np.zeros((noutputs, max_units))
output_prev_slopes = np.zeros((noutputs, max_units))

# For each candidate unit, create the vectors holding the correlations, incoming weights, and other stats
cand_cor = np.zeros((ncandidates, noutputs))
cand_prev_cor = np.zeros((ncandidates, noutputs))
cand_weights = np.zeros((ncandidates, max_units+1))
cand_deltas = np.zeros((ncandidates, max_units+1))
cand_slopes = np.zeros((ncandidates, max_units+1))
cand_prev_slopes = np.zeros((ncandidates, max_units+1))
cand_derivs = np.zeros((ncandidates, max_units+1))


unit_type = SigmoidHiddenUnit()
output_type = SigmoidOutputUnit()
dataloader = Dataloader(training_inputs, training_outputs, use_training_breaks,
                 training_breaks, test_inputs, test_outputs, use_test_breaks, test_breaks)

network = CascorNetwork(ncandidates, unit_type, output_type, use_cache, score_threshold, dataloader, raw_error,
                        hyper_error,
                 noutputs, ninputs, max_units, distribution=np.random.uniform)
stats = CascorStats()
candidate_trainer = CandidateUnitTrainer(network, input_patience, input_change_threshold, input_shrink_factor,
                 input_mu, input_decay, input_epsilon, stats)

outlimit= 100
inlimit = 100
rounds = 100

ctrainer = CascorTrainer(network, candidate_trainer, outlimit, inlimit, rounds, output_patience, output_epsilon,
                 output_mu, output_decay, output_deltas, output_slopes, output_prev_slopes, output_shrink_factor,
                 output_change_threshold, stats, weight_multiplier=1, test_function=None, test=False, restart=False)

ctrainer.train()



