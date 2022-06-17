#!/bin/bash
set -e
DATASET=$1

curl https://ai-s2-cascader.s3.us-west-2.amazonaws.com/${DATASET}.zip -o out/${DATASET}/models.zip
cd out/${DATASET}/
unzip models.zip
mv ${DATASET}/* .
rm models.zip
rm -r ${DATASET}/
