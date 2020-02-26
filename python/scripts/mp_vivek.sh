#!/bin/bash

python compute_mps.py \
    -i /home/mn2822/Desktop/WormTraces/data/Vivek/1010_tail_03/data.mat \
    -o /home/mn2822/Desktop/WormOT/data/vivek/1010_tail_03/mp_components/mp_0000_0900.mat \
    -d 'vivek' \
    -p 12 \
    -n 200 \
    -s 0 \
    -e 900
