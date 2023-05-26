#!/bin/bash

STATS_PATH=stats_ablation_models.txt

for NLAYERS in {1..12}
do 
    python sample_selector.py 100 50\% --nlayers $NLAYERS --output_file $STATS_PATH
done

for NLAYERS in {1..12}
do 
    python sample_selector.py 100\% --nlayers $NLAYERS --output_file $STATS_PATH
done