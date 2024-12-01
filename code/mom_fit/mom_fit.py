"""
Can we match 20 moments??
"""
import torch
import pandas as pd
from matching import *
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
from utils_sample_ph import *
import pickle as pkl
import time


def get_feasible_moments(original_size, n):
    """ Compute feasible k-moments by sampling from high order PH and scaling """
    k = original_size
    ps = torch.randn(k, k)
    lambdas = torch.rand(k) * 100
    alpha = torch.randn(k)
    a, T = make_ph(lambdas, ps, alpha, k)

    # Compute mean
    ms = compute_moments(a, T, k, 1)
    m1 = torch.stack(list(ms)).item()

    # Scale
    T = T * m1
    ms = compute_moments(a, T, k, n)
    momenets = torch.stack(list(ms))
    return momenets


if __name__ == "__main__":
    # orig_size = 50   # This is the size of the PH the moments come from (so we know they are feasible)

    for ind in range(250):

        use_size = np.random.randint(5,200)    # This is the size of the target PH
        n = np.random.randint(10, 21)           # This is the number of moments to match
        num_run = np.random.randint(1,10000)
        orig_size = np.random.randint(5, 150)
        a, A, moms = sample(orig_size)
        moms = compute_first_n_moments(a, A, n)
        ms = torch.tensor(np.array(moms).flatten())
        # ms = get_feasible_moments(original_size=orig_size, n=n)
        print(ms)

        ws = ms ** (-1)
        num_epochs = 300000
        (lambdas, ps, alpha), (a, T) = fit_ph_distribution(ms, use_size, num_epochs=num_epochs, moment_weights=ws)

        original_moments = ms.detach().numpy()
        computed_moments = [m.detach().item() for m in compute_moments(a, T, use_size, n)]
        moment_table = pd.DataFrame([computed_moments, original_moments], index="computed target".split()).T
        moment_table["delta"] = moment_table["computed"] - moment_table["target"]
        moment_table["delta-relative"] = moment_table["delta"] / moment_table["target"]
        print(moment_table)
        path  = '/scratch/eliransc/mom_match'
        file_name = 'num_run_' + str(num_run) + '_num_moms_'+str(n)+ '_orig_size_'+ str(orig_size)+'_use_size_'+str(use_size)+'_epochs_'+str(num_epochs)+'.pkl'
        full_path = os.path.join(path, file_name)
        pkl.dump(moment_table, open(full_path, 'wb'))







