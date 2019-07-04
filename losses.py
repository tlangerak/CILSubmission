'''
file for custom losses.
Only contains jaccard for now.
'''

import torch


def jaccard(target, predict):
    # Want to maximize jaccard, pytorch minimizes hence -1
    # optimal jaccard = 1 (or -1 in our implementation). This means that the target_false and predict_false and true have the same number of entries.
    # https://en.wikipedia.org/wiki/Jaccard_index (see figure with stopsign)
    N = target.size()[0]
    jac = (1 / N) * torch.sum((target * predict) / (target + predict - target * predict +1e-6))
    return -1 * jac
