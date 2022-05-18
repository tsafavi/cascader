import os
import yaml

from copy import deepcopy


class Config(object):
    """Nested dictionary-like option for getting and setting job options"""

    JOB_DEFAULTS = {
        "do-logging": False,
        "do-checkpoint": False,
        "job-modes": ["train", "test"],
        "debug-mode": False,
        "progress-bar-refresh-rate": 20,
        "num-sanity-val-steps": 0,
        "profiler-type": None,
        "num-gpus": 1,
    }

    DEBUG_DEFAULTS = {
        "debug.limit-train-batches": 50,
        "debug.limit-val-batches": 50,
    }

    MONITOR_DEFAULTS = {
        "monitor.metric": "mrr",
        "monitor.mode": "max",  # one of 'min', 'max'
    }

    SEARCH_DEFAULTS = {
        "search.num_trials": 10,
    }

    STOPPING_DEFAULTS = {
        "early_stopping.min_delta": 0.00,
        "early_stopping.patience": 5,
        "early_stopping.verbose": False,
    }

    DATASET_DEFAULTS = {
        "dataset.base": "data",
        "dataset.splits.train": "train",
        "dataset.splits.valid": "valid",
        "dataset.splits.test": ["test"],
        "dataset.description_col": "description",
        "dataset.tokenize_relations": True,
        "dataset.text.entity_filename": "entity_text.tsv",
        "dataset.text.subj_repr": ["name", "description"],
        "dataset.text.obj_repr": ["name", "description"],
        "dataset.text.relation_filename": "relation_text.tsv",
    }

    TRAIN_DEFAULTS = {
        "train.batch_size": 16,
        "train.max_epochs": 10,
        "train.lr": 2e-5,
        "train.warmup_frac": 0,
        "train.use_bce_loss": True,
        "train.use_margin_loss": True,
        "train.use_relation_cls_loss": True,
        "train.relation_cls_loss_weight": 0.25,
        "train.margin": 1,
        "train.negative_samples.num_neg_per_pos": 1,
    }

    EVAL_DEFAULTS = {
        "eval.batch_size": 16,
        "eval.checkpoint_path": "",
        "eval.check_val_every_n": 10,
        "eval.save_scores": True,
    }

    LM_DEFAULTS = {
        "lm.model_name": "bert-base-uncased",
        "lm.checkpoint_path": "",
        "lm.max_length": 128,
        "lm.pool_strategy": "cls",
    }

    ANSWER_SELECTOR_DEFAULTS = {
        "answer_selector.quantiles": [0.5, 0.75, 0.9, 0.95],
        "answer_selector.mlp.hidden_channels": 64,
        "answer_selector.mlp.dropout": 0.1,
    }

    ENSEMBLE_DEFAULTS = {
        "ensemble.answer_selector_type": "static",
        "ensemble.quantile": 0.95,
        "ensemble.top_k": None,
    }

    def __init__(self, options, folder, delim="."):
        """This shouldn't be called directly by the user"""
        self.options = options
        self.folder = folder
        self.delim = delim

        DEFAULT_KEY_VALS = [
            Config.JOB_DEFAULTS,
            Config.DEBUG_DEFAULTS,
            Config.MONITOR_DEFAULTS,
            Config.SEARCH_DEFAULTS,
            Config.STOPPING_DEFAULTS,
            Config.DATASET_DEFAULTS,
            Config.TRAIN_DEFAULTS,
            Config.EVAL_DEFAULTS,
            Config.LM_DEFAULTS,
            Config.ANSWER_SELECTOR_DEFAULTS,
            Config.ENSEMBLE_DEFAULTS,
        ]

        for defaults in DEFAULT_KEY_VALS:
            for key, value in defaults.items():
                if not self.has_key(key):
                    self.set(key, value)

    @staticmethod
    def create_from_yaml(path, delim="."):
        """Create a config from a YAML file"""
        with open(path) as f:
            options = yaml.load(f, Loader=yaml.SafeLoader)
            return Config(options, os.path.dirname(path), delim=delim)

    @staticmethod
    def create_from_config(config, folder=None):
        """Create a config by copying another Config object"""
        if folder is None:
            folder = config.folder
        return Config(deepcopy(config.options), folder, delim=config.delim)

    def save_as_yaml(self, fname="config.yaml"):
        fpath = os.path.join(self.folder, fname)
        with open(fpath, "w") as f:
            yaml.dump(self.options, f)

    def create_from_key(self, key):
        """Create a new config given a nested key"""
        return Config(self.get(key), self.folder, delim=self.delim)

    def to_dict(self):
        """Return the configuration as a nested dictionary"""
        return deepcopy(self.options)

    def to_flat_dict(self):
        """Return the configuration as a single-level dictionary"""
        options = self.to_pandas()
        return options.to_dict(orient="records")[0]

    def get(self, fullkey, default=None):
        """Get the value of a nested key expressed as a delimited string"""
        keys = fullkey.split(self.delim)
        values = self.options

        for key in keys:
            if not isinstance(values, dict) or key not in values:
                return default
            values = values[key]
        return values

    def set(self, fullkey, value):
        """Set the value of a nested key expressed as a delimited string"""
        keys = fullkey.split(self.delim)
        values = self.options

        for key in keys[:-1]:
            if key not in values:
                values[key] = {}
            values = values[key]

        values[keys[-1]] = value
        return True

    def has_key(self, fullkey):
        """Return true if this key exists in the config"""
        keys = fullkey.split(self.delim)
        values = self.options

        for key in keys:
            if not isinstance(values, dict) or key not in values:
                return False
            values = values[key]
        return True
