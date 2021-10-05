#!/bin/bash
# make sure you run this script from the root folder
#  using the following command
# $ bash evaluation/evaluate_dev.sh
python3 evaluation/submission_scorer.py -o data/system_output/dev_system_output.tsv -l data/tc4tl_data_v5/tc4tl/docs/tc4tl_dev_trials.tsv -r data/tc4tl_data_v5/tc4tl/docs/tc4tl_dev_key.tsv
