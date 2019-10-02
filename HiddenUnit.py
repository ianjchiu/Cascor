class HiddenUnit():
    def __init__(self):
        pass


    @staticmethod
    def activation_singleton(acc_sum):
        """Given the sum of weighted inputs, compute the unit's activation value."""
        pass


    @staticmethod
    def activation(acc_sum):
        """Given a vector of the units' sum of weighted inputs, compute the unit's activation value."""
        pass


    @staticmethod
    def activation_prime(acc_sum):
        """Given a vector of units' activation value and sum of weighted inputs, compute
         the derivative of the activation with respect to the sum. """
        pass

class GaussianHiddenUnit(HiddenUnit):
    def __init__(self):
        pass

    def activation_singleton(acc_sum):

        global unit_type
        if unit_type == "sigmoid":
            if acc_sum < -15.0:
                return -0.5
            elif acc_sum > 15:
                return 0.5
            else:
                return (1.0 / (1.0 + exp(-acc_sum))) - 0.5


    def activation(acc_sum):
        with np.errstate(divide='ignore', over='ignore', under='ignore'):
            return torch.where((torch.abs(acc_sum) <= 15), (1.0 / (1.0 + torch.exp(-acc_sum))) - 0.5, torch.sign(
                acc_sum) * 0.5)
