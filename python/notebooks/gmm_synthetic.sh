#!/bin/bash

python get_gmms.py \
    -i /home/mn2822/Desktop/WormOT/data/synthetic/fast_3d/gmm_data_3d.h5 \
    -o /home/mn2822/Desktop/WormOT/data/synthetic/fast_3d/syn_data_mp_test.mat \
    -d 'synthetic' \
    -p 12 \
    -n 10
