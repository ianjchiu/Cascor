from datasets import morse as ds
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
from CascorTrainer import CascorTrainer



start_seconds = None
finished_seconds = None


# Assorted Parameters and Controls.

# These parameters and switches control the quickprop learning algorithm
# used to train the output weights and candidate units.


unit_type = "sigmoid"
"""The type of activation function used by the hidden units.  Options
  currently implemented are sigmoid, asigmoid, and gaussian.  Sigmoid is
  symmetric in range -0.5 to +0.5, while Asigmoid is asymmetric, 0.0 to
  1.0."""

output_type = "sigmoid"
"""The activation function to use on the output units.  Options currently
  implemented are linear and sigmoid."""

hyper_error = False
"""If True, use hyperbolic arctan error"""

raw_error = False
"""If True, do not scale error by derivative of the output unit's activation function."""

sigmoid_prime_offset = 0.1
"""This is added to the derivative of the sigmoid function to prevent the
  system from getting stuck at the points where sigmoid-prime goes to
  zero."""

weight_range = 1.0
"""Input weights in the network get initial random values between plus and
  minus weight_range.  This parameter also controls the initial weights
  on direct input-to-output links."""



distribution = torch.distributions.uniform.Uniform(-weight_range, weight_range)
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
"""Vector of input patterns for training the net."""


training_outputs = ds.training_outputs
"""Vector of output patterns for training the net."""

use_training_breaks = ds.use_training_breaks
"""If true, use the training breaks vector.  Else, never break."""

training_breaks = ds.training_breaks
"""If use_training_breaks is true, this must be a simple vector with
one entry, True or False, for each training case.  If True, zero all accumulated
state before processing this case."""

# Test inputs and outputs are just the same in form as training inputs#
# and outputs.  If there is to be no distinct test set, just set
# *test-inputs* EQ to *training-inputs*.

test_inputs = ds.test_inputs
"""Vector of input patterns for testing the net."""

test_outputs = ds.test_outputs
""""Vector of output patterns for testing the net."""

use_test_breaks = ds.use_test_breaks
"""If true, use the test breaks vector.  Else, never break."""

test_breaks = ds.training_breaks
"""If use_test_breaks is true, this must be a simple vector with
one entry, T or NIL, for each test case.  If True, zero all accumulated
state before processing this case."""

# Some other data structures and parameters that control training.

max_cases = 0
"""Maximum number of training cases that can be accommodated by the current
  data structures."""


ncases = 0
"""Number of training cases currently in use.  Assume a contiguous block
  beginning with first_case."""


first_case = 0
"""Address of the first training case in the currently active set.  Usually
  zero, but may differ if we are training on different chunks of the training
  set at different times."""

# For some benchmarks there is a separate set of values used for testing
# the network's ability to generalize.  These values are not used during
# training.


# Fundamental data structures.

# Unit values and weights are floats.

# Instead of representing each unit by a structure, we represent the
# unit by an int.  This is used to index into various vectors that hold
# per-unit information, such as the activation value of each unit.
# So the information concerning a given unit is found in a slice of values
# across many vectors, all with the same unit-index.

# Per-connection information for each connection COMING INTO a unit is
# stored in a vector of vectors.  The outer vector is indexed by the unit
# number, and the inner vector is then indexed by connection number.
# This is a sleazy way of implementing a 2-D array, faster in most Lisp
# systems than multiplying to do the index arithmetic, and more efficient
# if the units are sparsely connected.

# In this version, we assume that each unit gets input from all previous Unit 0, the "bias unit" is always at a
# maximum-on value.  Next come some input units, in order.  Unit 0, the "bias unit" is always at a maximum-on
# value. Next come some input "units", then some hidden units.  The final incoming weight is the recurrent
# self-connection.


# Output units have their own separate set of data structures and
# indices.  The units and outputs together form the "active" network.
# There are also separate data structures and indices for the "candidate"
# units that have not yet been added to the network.


max_units = 100
"""Maximum number of input values and hidden units in the network."""

ninputs = ds.ninputs
"""Number of inputs for this problem."""

noutputs = ds.noutputs
"""Number of outputs for this problem."""

nunits = 0
"""Current number of active units in the network.  This count includes all external
  inputs to the network and the bias unit."""

ncandidates = 16
"""Number of candidate units whose inputs will be trained at once."""

# The following vectors hold values related to hidden units in the active
# net and their input weights.  The vectors are created by BUILD_NET, after
# the dimension variables have been set up.

values = None
"""Vector holding the current activation value for each unit and input in
  the active net."""

values_cache = None
"""Holds a distinct values vector for each of the max_cases training
  cases.  Once we have computed the values vector for each training case,
  we can use it repeatedly until the weights or training cases change."""

extra_values = None
"""Extra values vector to use when not using the cache."""

weights = None
"""Vector of vectors with structure parallel to the connections vector.
  Each entry gives the weight associated with an incoming connection."""

# The following vectors hold values for the outputs of the active
# network and the output_side weights.

outputs = None
"""Vector holding the network output values."""

errors = None
"""Vector holding the current error value for each output."""

errors_cache = None
"""Holds a distinct errors vector for each of the  max_cases training
  cases.  Once we have computed the errors vector for a given training
  case, we can use it repeatedly until the weights of the training cases
  change."""

extra_errors = None
"""Extra errors vector to use when not using the cache."""

sum_errors = None
"""Accumulate for each output the sum of the error values over a whole
  output training epoch."""

dummy_sum_errors = None
"""Replace sum_errors with this during test epochs."""

output_weights = None
"""Vector of vectors.  For each output, we have a vector of output weights
  coming from the unit indicated by the index."""

output_deltas = None
"""Vector of vectors, parallel with output weights.  Each entry is the
  amount by which the corresponding output weight was changed last time."""

output_slopes = None
"""Vector of vectors, parallel with output weights.  Each entry is the
  partial derivative of the total error with respect to the corresponding
  weight."""

output_prev_slopes = None
"""Vector of vectors, parallel with output weights.  Each entry is the
  previous value of the corresponding  output_slopes entry."""

output_weights_record = None
"""The vector of output weights is recorded here after each output_training
  phase and just prior to the addition of the next unit.  This record
  allows us to reconstruct the network's performance at each of these
  points in time."""

# The following vectors have one entry for each candidate unit in the
# pool of trainees.

cand_scores = None
"""A vector holding the correlation score for each candidate unit."""

cand_values = None
"""A vector holding the most recent output value for each candidate unit."""

cand_sum_values = None
"""For each candidate unit, the sum of its values over an entire
  training set."""

cand_cor = None
"""A vector with one entry for each candidate unit.  This entry is a vector
  that holds the correlation between this unit's value and the residual
  error at each of the outputs, computed over a whole epoch."""

cand_prev_cor = None
"""Holds the  cand_cor values computed in the previous candidate training
  epoch."""

cand_weights = None
"""A vector with one entry for each candidate unit.  This entry is a vector
  that holds the current input weights for that candidate unit."""

cand_deltas = None
"""A vector with one entry for each candidate unit.  This entry is a vector
  that holds the input weights deltas for that candidate unit."""

cand_slopes = None
"""A vector with one entry for each candidate unit.  This entry is a vector
  that holds the input weights slopes for that candidate unit."""

cand_prev_slopes = None
"""A vector with one entry for each candidate unit.  This entry is a vector
  that holds the previous values of the input weight slopes for that
  candidate unit."""

cand_derivs = None
"""A vector of vectors, parallel in structure to the cand_weights vector.
For each weight wi, remember dV/dwi on the previous training case."""


unit_type = SigmoidHiddenUnit()
output_type = SigmoidOutputUnit()
dataloader = Dataloader(training_inputs, training_outputs, use_training_breaks,
                 training_breaks, test_inputs, test_outputs, use_test_breaks, test_breaks)

network = CascorNetwork(unit_type, output_type, use_cache, score_threshold, dataloader, raw_error, hyper_error,
                 noutputs, ninputs, max_units, distribution=torch.distributions.uniform.Uniform(-1, 1))

