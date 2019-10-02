# Cascade Correlation Algorithm

An implementation of recurrent Cascade-Correlation (cascor) in both NumPy and PyTorch. The goal of the project is to create a modern, user-friendly implementation of recurrent cascor. Additionally, we extend this project to PyTorch in order to explore the possibilities of utilizing GPUs for increased speed for recurrent cascor.

## Overview:
In both `Cascor-NumPy/` and `Cascor-PyTorch/`, the logic is identical. Therefore this overview will briefly cover the structure of the code to facilitate usage and experimentation for the project. 

### CascorNetwork.py
In CascorNetwork.py, I created a class for the recurrent cascor network. The purpose is to separate out the structure of the code so that training the network will be separated from utilizing the network. For Cascor, the key details are the hyperparameters of the network:
* weight_range - the possible range for the weights in the network
* ncandidates - the number of candidates in the candidate pool
* raw_error - True, if we do not want to scale the error by the derivative of the output's activation function
* hyper_error - If we use hyperbolic artan error or not
* score_threshold - How close the output needs to be to be counted as correct
* use_cache - If we cache forward-pass values instead of recomputing them all the time
* output_type - Output unit type
* ninputs - Number of inputs
* noutputs - Number of outputs
* dataloader - Contains the data for training / testing
* max_units - Maximum number of units permitted in the network

It also contains the network represented in the arrays:
* weights - array of weights from unit to unit
* outputs - stored outputs after a forward pass

### CandidateUnitTrainer.py
This class does one of the two major parts of training cascor. This is the input-forward pass training where we train our candidate units and find the candidate whose score correlates most strongly with the current error signal. Contains hyperparameters for candidate pool training:
* mu - Parameter for quickprop
* epsilon - The amount of linear gradient descent used to update unit input weights
* shrink_factor - Check if step size is too large
* decay - Keeps weights from growing too big
* patience - Number of allowed consecutive epochs without significant change
* change_threshold - Amount changed required to count as a significant change

### CascorTrainer.py
This class does most of the heavy lifting. It performs the outer loop of training the output weights after adding a new unit to the network. Additionally, it calls the candidate pool trainer to train the network to completion. Its hyperparameters of mu, epsilon, shrink_factor, decay, patience, change_threshold defined as above. The additional hyperparameters are:
* stats - Keeps track of the epoch and other statistics for the network
* outlimit - Upper limit on the number of cycles in output phase
* inlimit - Upper limit on the number of cycles in input phase (candidate unit training)
* rounds - Upper limit on number of unit-installation cycles


### Motivations and Future Extensions:
In tester.py, I hacked together a slight proof of correctness to make sure that the network ran the same as it did in the base code. With regards to separating HiddenUnit and OutputUnit from the network definition and the trainer definitions, this would allow for a natural extension where we used mixed units in the candidate pool. An example of testing the code would be in `tester.py` with the following commands:
```
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
``` 

In this case, we create our network by initializing the unit_type, output_type, and dataloader, and we call the constructor. In tester.py we are evaluating the network on Morse code, as done in Fahlman's paper on RCC.
