#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:51:27 2020

@author: Omid Sadjadi <omid.sadjadi@nist.gov>
"""


import argparse
import numpy as np
from submission_validator import validate_me


def score_me(reference, system_output, D, w_miss=1, w_fa=1):

    key = np.genfromtxt(reference, names=True, delimiter='\t',
                        encoding='ascii', dtype=None)
    out = np.genfromtxt(system_output, names=True, delimiter='\t',
                        encoding='ascii', dtype=None)

    ndcf = {}
    for subset, dists in D.items():
        if subset == 'coarse_grain':
            set_mask = (key[subset] == 'Y')
        else:
            set_mask = (key['coarse_grain'] == 'N')
        set_out = out[set_mask]
        set_key = key[set_mask]

        for d in dists:

            set_mask_contact = set_key['distance_in_meters'] <= d
            n_miss = sum(set_mask_contact & (set_out['distance'] > d))
            p_miss = n_miss/sum(set_mask_contact)

            set_mask_noncontact = set_key['distance_in_meters'] > d
            n_fa = sum(set_mask_noncontact & (set_out['distance'] <= d))
            p_fa = n_fa/sum(set_mask_noncontact)

            normalized_dcf = (w_miss * p_miss + w_fa * p_fa)/min(w_miss, w_fa)
            if subset in ndcf:
                ndcf[subset].append([d, p_miss, p_fa, normalized_dcf])
            else:
                ndcf[subset] = [[d, p_miss, p_fa, normalized_dcf]]

    return ndcf


def main():
    parser = argparse.ArgumentParser(description='TC4TL Submission Scorer.')
    parser.add_argument("-o", "--output", help="path to system output file",
                        type=str, required=True)
    parser.add_argument("-l", "--trials", help="path to the list of trials, "
                        "e.g., /path/to/tc4tl_dev_trials.tsv",
                        type=str, required=True)
    parser.add_argument("-r", "--reference", help="path to the reference key "
                        "for the trials, e.g., /path/to/tc4tl_dev_key.tsv",
                        type=str, required=True)

    args = parser.parse_args()
    system_output = args.output
    trials_list = args.trials
    reference = args.reference

    if validate_me(system_output, trials_list):
        print('system output failed the validation step!')
        return

    D = {'fine_grain': [1.2, 1.8, 3.0],
         'coarse_grain': [1.8]
         }

    ndcf = score_me(reference, system_output, D)
    print(f"Subset\t\tD\tP_miss\tP_fa\tnDCF")
    for subset, subset_scores in ndcf.items():
        for scores in subset_scores:
            results = '\t'.join([f'{sc:.2f}' for sc in scores])
            print(f"{subset}\t{results}")


if __name__ == '__main__':
    main()
