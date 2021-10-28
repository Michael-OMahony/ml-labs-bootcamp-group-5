import numpy as np
import pandas as pd

DEV_TRIALS_FP = "data/tc4tl_data_v5/tc4tl/docs/tc4tl_dev_trials.tsv"
TEST_TRIALS_FP = "data/tc4tl_data_v5/tc4tl/docs/tc4tl_test_trials.tsv"
DEV_KEY_FP = "data/tc4tl_data_v5/tc4tl/docs/tc4tl_dev_key.tsv"
TEST_KEY_FP = "data/tc4tl_test_key/tc4tl/docs/tc4tl_test_key.tsv"


def validate_me(system_output, trials_list, max_lines=20, verbose=False):
    invalid = False
    line_counter = 0
    err_str = ''
#    with open(trials_list) as fid1, open(system_output) as fid2:
    fid1 = open(trials_list)
    fid2 = open(system_output)
    line_no = 0
    for line in fid1:
        line_no += 1
        ref_list = split_line(line)
        sys_list = split_line(fid2.readline())
        # checking if the number of lines in two files match
        if sys_list == ['']:
            err_str += ('The system output has less lines than the trial '
                        'list.')
            invalid = True
            break
        # checking if the delimiter is TAB
        if len(sys_list) != len(ref_list) + 1:
            err_str += ('Line {}: Incorrect number of columns/fields. '
                        'Expected {}, got {} instead. TAB (\\t) delimiter '
                        'should be used.\n'.format(line_no,
                                                   len(ref_list)+1,
                                                   len(sys_list)))
            invalid = True
            line_counter += 1
        else:
            # checking if the fields match the reference
            if sys_list[:-1] != ref_list:
                err_str += ('Line {}: Incorrect field(s). Expected "{}", '
                            'got "{}" instead.\n'
                            .format(line_no, '\t'.join(ref_list),
                                    '\t'.join(sys_list[:-1])))
                invalid = True
                line_counter += 1
            if line_no == 1:
                # checking if "LLR" is in the header
                if sys_list[-1] != 'distance' or sys_list[0] != 'fileid':
                    err_str += ('Line {}: Expected "fileid<TAB>distance" '
                                'in the header, got "{}" instead.\n'
                                .format(line_no, '\t'.join(sys_list)))
                    invalid = True
                    line_counter += 1
            else:
                # checking if the scores are floats
                if not is_float(sys_list[-1]):
                    err_str += ('Line {}: Expected float in the distance '
                                'column, got "{}" instead.\n'
                                .format(line_no, sys_list[-1]))
                    invalid = True
                    line_counter += 1
        if line_counter >= max_lines:
            break
    ref_list = fid1.readline()
    sys_list = fid2.readline()
    # checking if the number of lines in two files match
    if sys_list and not ref_list:
        err_str += ('The system output has more lines than the trial list.')
        invalid = True
    fid1.close()
    fid2.close()
    if err_str and invalid:
        if verbose:
            print("\n" + err_str)
    return invalid


def split_line(line, delimiter='\t'):
    return line.strip().split(delimiter)


def is_float(astr):
    try:
        float(astr)
        return True
    except ValueError:
        return False


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


def scores_to_df(scores):
    rows = []
    for key, values in scores.items():
        for item in values:
            rows.append({
                'Subset': key,
                'Distance': item[0],
                'P_miss': item[1],
                'P_fa': item[2],
                'nDCF': item[-1]
            })
    return pd.DataFrame(rows)


def compute_ndcf(sysout):
    D = {
        'fine_grain': [1.2, 1.8, 3.0],
        'coarse_grain': [1.8]
    }
    # save sysout dataframe to disk
    sysout_fp = "data/system_output/temp.tsv"
    sysout.to_csv(sysout_fp, sep="\t", index=False)
    # validation
    if validate_me(sysout_fp, TEST_TRIALS_FP):
        # Development set
        assert not validate_me(sysout_fp, DEV_TRIALS_FP)
        scores = score_me(DEV_KEY_FP, sysout_fp, D)
        return scores_to_df(scores)
    # Test set
    scores = score_me(TEST_KEY_FP, sysout_fp, D)
    return scores_to_df(scores)
