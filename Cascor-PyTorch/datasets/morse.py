import random
import numpy as np

import torch

nletters = 26
running_output = False
code_conversions = [None] * 27

code_conversions[1] = ".-"    # A      2      4 (5)
code_conversions[2] = "-..."  # B      4      8 (9)
code_conversions[3] = "-.-."  # C      4      9 (11)
code_conversions[4] = "-.."   # D      3      6 (7)
code_conversions[5] = "."     # E      1      1 (1)
code_conversions[6] = "..-."  # F      4      8 (9)
code_conversions[7] = "--."   # G      3      7 (9)
code_conversions[8] = "...."  # H      4      7 (7)
code_conversions[9] = ".."    # I      2      3 (3)
code_conversions[10] = ".---" # J      4     10 (13)
code_conversions[11] = "-.-"  # K      3      7 (9)
code_conversions[12] = ".-.." # L      4      8 (9)
code_conversions[13] = "--"   # M      2      5 (7)
code_conversions[14] = "-."   # N      2      4 (5)
code_conversions[15] = "---"  # O      3      8 (11)
code_conversions[16] = ".--." # P      4      9 (11)
code_conversions[17] = "--.-" # Q      4     10 (13)
code_conversions[18] = ".-."  # R      3      6 (7)
code_conversions[19] = "..."  # S      3      5 (5)
code_conversions[20] = "-"    # T      1      2 (3)
code_conversions[21] = "..-"  # U      3      6 (7)
code_conversions[22] = "...-" # V      4      8 (9)
code_conversions[23] = ".--"  # W      3      7 (9)
code_conversions[24] = "-..-" # X      4      9 (11)
code_conversions[25] = "-.--" # Y      4     10 (13)
code_conversions[26] = "--.." # Z      4      9 (11)


def random_string(length, chars_used="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Returns a random string of length length using only the characters present
    in the string chars_used; removes duplicates so each character has equal
    probability of being chosen"""
    chars_used = list(set(chars_used.upper()))
    s = [None] * length
    for i in range(length):
        s[i] = random.choice(chars_used)
    return ''.join(s)


def string_to_code(s):
    """Given a string of characters, S returns three values: a vector of input
  patterns in the proper morse-code representation, a corresponding vector
  of output patterns, and a vector of break values with a T at the start of
  each new character."""
    inlist = []
    outlist = []
    breaklist = []
    for i in range(len(s)):
        c = ord(s[i]) - 64
        morse = code_conversions[c]
        outpat = [-0.5] * (nletters + 1)
        strobepat = [-0.5] * (nletters + 1)
        if (running_output):
            outpat[c] = 0.5
        strobepat[0] = 0.5
        strobepat[c] = 0.5
        for j in range(len(morse)):
            if (morse[j] == '.'):
                inlist = [[-0.5], [0.5]] + inlist
                outlist = [outpat[:], outpat[:]] + outlist
                breaklist = [False, (j == 0)] + breaklist
            else:
                inlist = [[-0.5], [0.5], [0.5]] + inlist
                outlist = [outpat[:], outpat[:], outpat[:]] + outlist
                breaklist = [False, False, (j == 0)] + breaklist
        outlist = [strobepat[:]] + outlist
        inlist = [[-0.5]] + inlist
        breaklist = [False] + breaklist

    inlist.reverse()
    outlist.reverse()
    breaklist.reverse()
    return (inlist, outlist, breaklist)

training_string = None
test_string = None

def build_morse(s, continuing=False):
    """Given a string of characters, create a training set for the morse code
  representation of the string.  There is no test set.  If CONTINUE is on,
  make use of pre-existing hidden units."""
    global training_string, training_inputs, training_outputs, training_breaks, use_training_breaks, test_inputs, \
                                                                        test_outputs, test_breaks, test_string, \
                                                                        use_test_breaks, ninputs, noutputs, nletters

    training_string = s
    (inputs, outputs, breaks) = string_to_code(s)
    training_inputs = torch.FloatTensor(inputs)
    training_outputs = torch.FloatTensor(outputs)
    training_breaks = breaks
    use_training_breaks = True
    test_inputs = None
    test_outputs = None
    test_breaks = training_breaks
    test_string = None
    use_test_breaks = True
    ninputs = 1
    noutputs = nletters + 1
    if continuing:
        changed_training_set()
    # build_net(ninputs, noutputs)
    # init_net()
    print(f'Training on {s}')


build_morse("ABCDEFGHIJKLMNOPQRSTUVWXYZ")



