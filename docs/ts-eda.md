# Exploratory Time Series Data Analysis


Ask these questions.

1. Do you understand the problem statement?
2. Can you ELI5 the data generating experiment? Can the experiment be recreated based on your explanation?


EDA should be in sync with your problem statement and should align with your
solution approach. You don't wanna have a bunch of irrelevant plots.

## Problem Statement before looking at the data

**Contact Tracing**

Can you predict if a person A has been in contact with person B based
on the Bluetooth Low Energy (BLE) readings from their smartphones?

They are said to have made contact if they have spent more than 15 minutes
being < 2 meters near each other.

**But...**

RSSI values of bluetooth (low-energy) signals are pretty noisy estimator of the actual distance between the phones.

Sensitive to real-world conditions like,

- Where the phones are carried
- Body positions
- Physical barriers
- Multi-path environments

**So...**

Use of phone sensor data in addition to RSSI values could help in building a 
robust estimator of distance between phones.
