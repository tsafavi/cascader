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
This will set up the `data/repodb` directory, consisting of entity and relation ID files, entity and relation text files, and train/dev/testing triple files. 

## Jobs

All jobs are implemented using the [PyTorch Lightning](https://www.pytorchlightning.ai/) API.
To run a job, use the following command:
```
python src/main.py <path_to_config_file>
```

Each job requires a path to a YAML configuration file. 
The file `src/config.py` provides default configuration options. 
You can set or overwrite these options in individual config files. 
Here is an example of a config file that trains a cross-encoder BERT-Base LM on the CoDEx-S dataset and evaluates the model on the validation and test sets: 
```
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
  batch_size: 32
  max_epochs: 20
  use_bce_loss: True
  use_margin_loss: True
  use_relation_cls_loss: True
  lr: 2.0e-5
  margin: 1
  negative_samples:
    num_neg_per_pos: 2
  max_length: 256
  model_name: bert-base-uncased
eval:
  batch_size: 16
  check_val_every_n: 5
```

## Replicate experiments

We provide pretrained KGEs/LMs for each dataset at the following links:
- RepoDB
	- [KGE]()
	- [Dual-encoder]()
	- [Cross-encoder]()
- CoDEx-S
	- [KGE]()
	- [Dual-encoder]()
	- [Cross-encoder]()
- CoDEx-M
	- [KGE]()
	- [Dual-encoder]()
	- [Cross-encoder]()
- WN18RR
	- [KGE]()
	- [Dual-encoder]()
	- [Cross-encoder]()
- FB15K-237
	- [KGE]()
	- [Dual-encoder]()
	- [Cross-encoder]()




