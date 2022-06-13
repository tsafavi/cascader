#!/bin/bash
set -e
DATASET=$1

if [ "$DATASET" == "fb15k-237" ] || [ "$DATASET" == "wn18rr" ]; then
    echo "${DATASET} not yet uploaded to S3"
    exit 1
fi

curl https://ai-s2-cascader.s3.us-west-2.amazonaws.com/${DATASET}.zip -o out/${DATASET}/models.zip
cd out/${DATASET}/
unzip models.zip
mv ${DATASET}/* .
rm models.zip
rm -r ${DATASET}/