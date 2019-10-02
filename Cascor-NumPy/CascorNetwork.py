from __future__ import absolute_import, division, print_function
import torch


class CascorNetwork:
    """Class definition of Recurrent Cascor Network."""

    def __init__(self, ncandidates, unit_type, output_type, use_cache, score_threshold, dataloader, raw_error,
                 hyper_error, noutputs=1, ninputs=1, max_units=100,
                 distribution=np.random.uniform, weight_range=1):
        self.weight_range = weight_range
        self.ncandidates = ncandidates
        self.raw_error = raw_error
        self.hyper_error = hyper_error
        self.score_threshold = score_threshold
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
        self.extra_values = np.zeros(self.max_units)
        self.extra_values[0] = 1.0
        self.values = self.extra_values
        self.weights = np.zeros((self.max_units, self.max_units))
        self.outputs = np.zeros(self.noutputs)
        self.extra_errors = np.zeros(self.noutputs)
        self.errors = self.extra_errors
        self.sum_errors = np.zeros(self.noutputs)
        self.dummy_sum_errors = np.zeros(self.noutputs)

        if self.use_cache:
            self.values_cache = np.zeros((self.max_cases, self.max_units))
            self.errors_cache = np.zeros((self.max_cases, self.noutputs))
        # For each output, create the vectors holding per-weight info
        self.output_weights = np.zeros((self.noutputs, self.max_units))
        self.output_weights[:self.noutputs, :1+self.ninputs] = \
            self.distribution(self.noutputs, 1+self.ninputs) * 2 * self.weight_range - self.weight_range

    def set_up_inputs(self, input_vec):
        """Set up all the inputs from the input_vec vector as the first few entries in the values vector"""
        self.values[0] = 1.0
        self.values[1:self.ninputs+1] = input_vec

    def compute_unit_value(self, j, prev_value):
        """Assume that values vector has correct current values for all units with index less than J.
        Compute, record, and return the value for unit J. prev_values is is the previous value of unit J."""
        w = self.weights[j]
        self.values[j] = self.unit_type.activation_singleton(np.matmul((w[:j] * self.values[:j]), np.ones((j, 1))) +
                                                             prev_value * w[j])
        return self.values[j]

    def output_forward_pass(self):
        """Assume the values vector has been set up. Just compute the network's outputs"""
        self.outputs = np.matmul(self.values[:self.nunits],np.transpose(self.output_weights[:, :self.nunits]))
        self.outputs = np.array(list(map(self.output_type.output_function, self.outputs)))

    def recompute_errors(self, goal):
        """Like compute errors, but don't bother updating slopes and statistics."""
        for j in range(self.noutputs):
            out = self.outputs[j]
            diff = out - goal[j]
            err_prime = diff * self.output_type.output_prime(out)
            self.errors[j] = err_prime

    def full_forward_pass(self, input_vec, no_memory):
        """This is called only when not using the cache. Set up the inputs from the input_vec vector,
        then propagate activation values forward through all hidden units and output units.
        If no-memory is True, assume the previous unit values are all zero."""
        self.set_up_inputs(input_vec)
        for j in range(1 + self.ninputs, self.nunits):
            if no_memory:
                self.compute_unit_value(j, 0.0)
            else:
                self.compute_unit_value(j, self.values[j])
        self.output_forward_pass()

    def install_new_unit(self, unit, prev_cor, weight_multiplier):
        """Add the candidate-unit with the best correlation score to the active
          network."""
        if self.nunits >= self.max_units:
            return False
        self.weights[self.nunits, :self.nunits+1] = unit[:1+self.nunits]
        print("  Add unit {0}: {1}".format(self.nunits - self.ninputs, unit))
        self.output_weights[:, self.nunits] = weight_multiplier *  -prev_cor
        if self.use_cache:
            prev_value = 0.0
            for i in range(self.max_cases):
                self.values = self.values_cache[i]
                if self.dataloader.use_training_breaks and self.dataloader.training_breaks[i]:
                    prev_value = self.compute_unit_value(self.nunits, 0.0)
                else:
                    prev_value = self.compute_unit_value(self.nunits, prev_value)
        self.nunits += 1
        return True


