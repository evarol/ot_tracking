#!/bin/bash

python compute_mps.py \
    -i /home/mn2822/Desktop/WormOT/data/zimmer/raw/mCherry_v00065-00115.hdf5 \
    -o /home/mn2822/Desktop/WormOT/data/zimmer/mp_components/mp_0000_0050.mat \
    -d 'zimmer' \
    -p 12 \
    -n 200 \
    -s 0 \
    -e 50
