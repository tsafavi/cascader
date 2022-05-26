# README.md

This repository contains the data and PyTorch implementation of the arXiv submission 
_[CascadER: Cross-Modal Cascading for Knowledge Graph Link Prediction](https://arxiv.org/abs/2205.08012)_ by Tara Safavi, Doug Downey, and Tom Hope. 

If you use our work, please cite us as follows:
```
@article{safavi2022cascader,
  title={CascadER: Cross-Modal Cascading for Knowledge Graph Link Prediction},
  author={Safavi, Tara and Downey, Doug and Hope, Tom},
  journal={arXiv preprint arXiv:2205.08012},
  year={2022}
}
```

## Quick start

Run the following to set up your virtual environment and install the Python requirements: 
```
python3.7 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

To setup a dataset, e.g., RepoDB:
```
cd data
unzip repodb.zip
```
This will set up the `data/repodb/` directory, consisting of entity and relation ID files, entity and relation text files, and train/dev/testing triple files. 

## Download models
To download three pretrained models (KGE, dual-encoder, & cross-encoder) for a given dataset, use the following:
```
chmod u+x download_data.sh
./download_data.sh <dataset_name>
```
For example, the command `./download_data.sh repodb` will download the following files:
- `out/repodb/kge.ckpt`
- `out/repodb/biencoder.ckpt`
- `out/repodb/crossencoder.ckpt`

__Be aware that the model files are very large for the larger datasets, up to 7 GB for FB15K-237, because all of the query/answer score pairs for the validation/test sets are saved in these model files.__

## Run cascades with pretrained models

### No pruning
To run a full 3-stage cascade without any pruning, use the following:
```
chmod u+x cascade_full.sh
./cascade_full.sh <dataset_name>
```
- This will first run _Tier 1_ reranking (KGE + bi-encoder), searching over the optimal weighting of the two models' scores in 10 trials. 
The results of the best trial from _Tier 1_ will be saved to `out/<dataset_name>/t1/checkpoints/checkpoint_best.pt`.
- Next, this will run _Tier 2_ reranking (Tier 1 output + cross-encoder), again searching over the optimal weighting of the two sets of scores in 10 trials. The results of the best trial from _Tier 2_ will be saved to `out/<dataset_name>/t2/checkpoints/checkpoint_best.pt`.

### With pruning
To run a 3-stage cascade with pruning between _Tier 1_ and _Tier 2_, use the following:
```
chmod u+x cascade_pruned.sh
./cascade_pruned.sh <dataset_name>
```
- This will first run _Tier 1_ reranking (KGE + bi-encoder), searching over the optimal weighting of the two models' scores in 10 trials (same as above).
The results of the best trial from _Tier 1_ will be saved to `out/<dataset_name>/t1/checkpoints/checkpoint_best.pt`.
- Next, this will run an _Answer Selector_ job in which we predict the number of answers to rerank for reach query. The results of answer selection will be saved to `out/<dataset_name>/t1_prune/checkpoints/checkpoint_best.pt`.
- Finally, this will run pruned _Tier 2_ reranking (Tier 1 output + cross-encoder over _Answer Selector_ outputs only), again searching over the optimal weighting of the two sets of scores in 10 trials. The results of the best trial from pruned _Tier 2_ will be saved to `out/<dataset_name>/t2_prune/checkpoints/checkpoint_best.pt`.

## Run new jobs

All jobs are implemented using the [PyTorch Lightning](https://www.pytorchlightning.ai/) API.

To run a job, use the following command:
```
python src/main.py <path_to_config_file>
```

Each job requires a path to a YAML configuration file. 
The file `src/config.py` provides default configuration options for job outputs, model training hyperparameters, etc. 
You can set or overwrite these options in individual config files. 

### Training config example
Here is an example of a config file that trains a cross-encoder BERT-Base LM on the CoDEx-S dataset and evaluates the model on the validation and test sets: 
```
do-checkpoint: True  # by default False, set to True if you want to save model weights and ranking outputs
job-modes:
  - train  # remove if you want to evaluate the model only
  - test
dataset:
  name: codex-s  # if custom, you must provide the corresponding dataset in the data/ directory
  num_entities: 2034
  num_relations: 42
  text:
    subj_repr:  # concatenate ‘name’ and ‘extract’ columns from codex-s entity file for subject entity description
      - name
      - extract
    obj_repr:
      - name
      - extract
  splits:
    test:  # get model prediction scores on validation and test splits
      - valid
      - test
train:
  model_type: crossencoder
  batch_size: 16
  max_epochs: 5
  use_bce_loss: True
  use_margin_loss: True
  use_relation_cls_loss: True
  lr: 1.0e-5
  margin: 1
  negative_samples:
    num_neg_per_pos: 2
lm:
  model_name: bert-base-uncased
  max_length: 128
eval:
  batch_size: 16
  check_val_every_n: 5
```

### Model selection config example

To run a job and select a model over a specified set of hyperparameters, add the `--search` flag to your job invocation as follows:
```
python src/main.py <path_to_config_file> --search
```

Here is an example of a config file that trains a cross-encoder BERT-Base LM on the CoDEx-S dataset and evaluates the model on the validation and test sets, __searching over the optimal learning rate, margin, and number of negative samples in 5 trials__:
```
do-checkpoint: True  # by default False, set to True if you want to save model weights and ranking outputs
job-modes:
  - train  # remove if you want to evaluate the model only
  - test
dataset:
  name: codex-s  # if custom, you must provide the corresponding dataset in the data/ directory
  num_entities: 2034
  num_relations: 42
  text:
    subj_repr:  # concatenate ‘name’ and ‘extract’ columns from codex-s entity file for subject entity description
      - name
      - extract
    obj_repr:
      - name
      - extract
  splits:
    test:  # get model prediction scores on validation and test splits
      - valid
      - test
train:
  model_type: crossencoder
  batch_size: 16
  max_epochs: 5
  use_bce_loss: True
  use_margin_loss: True
  use_relation_cls_loss: True
  lr: 1.0e-5
  margin: 1
  negative_samples:
    num_neg_per_pos: 2
lm:
  model_name: bert-base-uncased
  max_length: 128
eval:
  batch_size: 16
  check_val_every_n: 5
search:
  num_trials: 5
  parameters:
  - name: train.lr
    type: choice
    value_type: float
    values:
    - 1e-5
    - 2e-5
    - 3e-5
  - name: train.margin
    type: range
    value_type: int
    bounds:
    - 1
    - 10
  - name: train.negative_samples.num_neg_per_pos
    type: range
    value_type: int
    bounds:
    - 1
    - 5
```

### Reranking config example

To run a reranking job over a pair of models and select optimal weights for the two models' scores, use the following: 
```
python src/main.py <path_to_config_file> --search
```

Here is an example of a reranking job that searches over the optimal additive ensemble between a KGE and a cross-encoder on CoDEx-S: 
```
do-checkpoint: True
job-modes:  # no training since base models are already trained
  - validate  # must include validation to select the optimal weights
  - test
dataset:
  name: codex-s
  num_entities: 2034
  num_relations: 42
train:
  model_type: ensemble
ensemble:
  base_ranker_checkpoint_path: out/codex-s/kge.ckpt
  reranker_checkpoint_path: out/codex-s/crossencoder.ckpt
search:
  parameters:
  - bounds:
    - 0.05
    - 0.95
    name: ensemble.reranker_weight_head_batch
    type: range
    value_type: float
  - bounds:
    - 0.05
    - 0.95
    name: ensemble.reranker_weight_tail_batch
    type: range
    value_type: float
```
