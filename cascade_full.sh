#!/bin/bash
set -e
DATASET=$1

if [ "$DATASET" == "fb15k-237" ] || [ "$DATASET" == "wn18rr" ]; then
    echo "full reranking not supported for ${DATASET}"
    exit 1
fi

python src/main.py ${DATASET}/t1/config.yaml --search  # run reranking at tier 1
python src/main.py ${DATASET}/t2/config.yaml --search  # run reranking at tier 2