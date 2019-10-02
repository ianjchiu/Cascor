from __future__ import absolute_import, division, print_function


class Dataloader:
    """Stores the training / test data required for training and evaluating the Cascor Model"""
    def __init__(self, training_inputs, training_outputs, use_training_breaks,
                 training_breaks, test_inputs, test_outputs, use_test_breaks, test_breaks):
        self.test_inputs = test_inputs
        self.training_inputs = training_inputs
        self.test_outputs = test_outputs
        self.training_outputs = training_outputs
        self.use_test_breaks = use_test_breaks
        self.use_training_breaks = use_training_breaks
        self.training_breaks = training_breaks
        self.test_breaks = test_breaks
