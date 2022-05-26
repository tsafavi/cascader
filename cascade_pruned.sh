#!/bin/bash
set -e
DATASET=$1

python src/main.py ${DATASET}/t1/config.yaml --search  # run reranking at tier 1
python src/main.py ${DATASET}/t1_prune/config.yaml  # run answer pruning at tier 1
python src/main.py ${DATASET}/t2_prune/config.yaml --search  # run reranking at tier 2