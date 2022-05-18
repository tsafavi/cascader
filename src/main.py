import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from config import Config
from data.biencoder_dataset import BiEncoderTrainDataset, BiEncoderEvalDataset
from data.crossencoder_dataset import CrossEncoderTrainDataset, CrossEncoderEvalDataset
from data.crossencoder_reranker_dataset import CrossEncoderRerankerEvalDataset
from data.answer_selector_dataset import AnswerSelectorDataset
from data.score_eval_dataset import ScoreEvalDataset
from data.reranker_dataset import RerankingDataset
from data.util import load_scores

from lightning_modules.biencoder import BiEncoder
from lightning_modules.crossencoder import CrossEncoder
from lightning_modules.crossencoder_reranker import CrossEncoderReranker
from lightning_modules.answer_selector import AnswerSelector
from lightning_modules.score_eval import QueryScoreEvalModule

from model_selection.ax import AxSearchJob

QUERY_MODES = ["head-batch", "tail-batch"]


def main(config):
    """Wrapper function for running train and/or test with PyTorch Lightning"""
    do_logging = config.get("do-logging")
    do_checkpoint = config.get("do-checkpoint")
    job_modes = config.get("job-modes")
    debug = config.get("debug-mode")
    profiler_type = config.get("profiler-type")
    progress_bar_refresh_rate = config.get("progress-bar-refresh-rate")
    num_sanity_val_steps = config.get("num-sanity-val-steps")

    if torch.cuda.device_count() > 0:
        num_gpus = config.get("num-gpus")
        device = "cuda"
    else:
        num_gpus = 0
        device = "cpu"

    # Initialize callbacks
    monitor_metric = config.get("monitor.metric")
    monitor_mode = config.get("monitor.mode")
    stopping_min_delta = config.get("early_stopping.min_delta")
    stopping_patience = config.get("early_stopping.patience")
    stopping_verbose = config.get("early_stopping.verbose")

    checkpoint_filename_format = "checkpoint_best"
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=stopping_min_delta,
        patience=stopping_patience,
        verbose=stopping_verbose,
        mode=monitor_mode,
        check_finite=True,
    )

    callbacks = [early_stopping_callback]

    if do_checkpoint:
        model_checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric,
            mode=monitor_mode,
            filename=checkpoint_filename_format,
        )
        callbacks.append(model_checkpoint_callback)
        checkpoint_callback = True
    else:
        checkpoint_callback = False

    # Initialize a trainer
    max_epochs = config.get("train.max_epochs")
    check_val_every_n_epoch = config.get("eval.check_val_every_n")

    if debug:
        limit_train_batches = config.get("debug.limit-train-batches")
        limit_val_batches = config.get("debug.limit-val-batches")
    else:
        limit_train_batches = 1.0  # use full train dataset
        limit_val_batches = 1.0  # use full validation dataset

    # Initialize logger
    default_root_dir = config.folder
    if do_logging:
        logger = pl_loggers.TensorBoardLogger(os.path.join(default_root_dir, "logs/"))
    else:
        logger = False

    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        gpus=num_gpus,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_val_batches,
        profiler=profiler_type,
        callbacks=callbacks,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=logger,
    )

    # Initialize a module and train/valid/test datasets
    model_type = config.get("train.model_type")
    batch_size = config.get("train.batch_size")
    eval_batch_size = config.get("eval.batch_size")
    test_split_names = config.get("dataset.splits.test")
    if not isinstance(test_split_names, list):
        test_split_names = [test_split_names]

    # Model factory
    if model_type == "biencoder":
        setup_dict = _setup_biencoder(config)
    elif model_type == "crossencoder":
        setup_dict = _setup_crossencoder(config)
    elif model_type == "crossencoder_reranker":
        setup_dict = _setup_crossencoder_reranker(config)
    elif model_type == "answer_selector":
        setup_dict = _setup_answer_selector(config)
    elif model_type == "kge_eval" or model_type == "score_eval":
        setup_dict = _setup_score_eval(config)
    else:  # set up cascade
        setup_dict = _setup_reranking_pipeline(config)

    pl_module = setup_dict["pl_module"]

    outputs = {}
    if "train" in job_modes or "validate" in job_modes:
        valid_datasets = setup_dict["valid_datasets"]
        eval_collate_fn = setup_dict["eval_collate_fn"]
        valid_loaders = {
            query_mode: DataLoader(
                valid_datasets[query_mode],
                batch_size=eval_batch_size,
                collate_fn=eval_collate_fn,
            )
            for query_mode in QUERY_MODES
        }

    # Training
    if "train" in job_modes:
        train_datasets = setup_dict["train_datasets"]
        train_collate_fn = setup_dict["train_collate_fn"]
        train_loaders = {
            query_mode: DataLoader(
                train_datasets[query_mode],
                shuffle=True,
                batch_size=batch_size,
                collate_fn=train_collate_fn,
            )
            for query_mode in QUERY_MODES
        }

        trainer.fit(
            model=pl_module,
            train_dataloaders=train_loaders,
            val_dataloaders=CombinedLoader(valid_loaders),
        )

    # Evaluation
    if do_checkpoint and model_checkpoint_callback.best_model_path != "":
        ckpt_path = model_checkpoint_callback.best_model_path
    elif config.get("eval.checkpoint_path") != "":
        ckpt_path = config.get("eval.checkpoint_path")
    else:
        ckpt_path = None

    if ckpt_path is not None:
        # Load the checkpoint file and initialize the Lightning module
        ckpt = torch.load(ckpt_path, map_location=device)
        pl_module = type(pl_module).load_from_checkpoint(
            checkpoint_path=ckpt_path, map_location=device
        )
    elif do_checkpoint:
        # Create a new checkpoint
        ckpt = {}
        ckpt_folder = os.path.join(config.folder, "checkpoints")
        if not os.path.isdir(ckpt_folder):
            os.makedirs(ckpt_folder, exist_ok=True)
        ckpt_path = os.path.join(ckpt_folder, checkpoint_filename_format + ".ckpt")
    else:
        ckpt = None

    if "validate" in job_modes:
        trainer.validate(model=pl_module, dataloaders=CombinedLoader(valid_loaders))
        valid_metrics = trainer.callback_metrics
        outputs.update(valid_metrics)

        if do_checkpoint and config.get("eval.save_scores"):
            ckpt = pl_module.on_save_checkpoint(ckpt)
            torch.save(ckpt, ckpt_path)

    if "test" in job_modes:
        test_datasets_by_split = setup_dict["test_datasets_by_split"]
        eval_collate_fn = setup_dict["eval_collate_fn"]

        test_loaders_by_split = {}  # {test_split_name: {query_mode: dataloader}}
        for test_split_name, test_datasets in test_datasets_by_split.items():
            test_loaders_by_split[test_split_name] = {
                query_mode: DataLoader(
                    test_datasets[query_mode],
                    batch_size=eval_batch_size,
                    collate_fn=eval_collate_fn,
                )
                for query_mode in QUERY_MODES
            }

        for test_split_name, test_loaders in test_loaders_by_split.items():
            trainer.test(model=pl_module, dataloaders=CombinedLoader(test_loaders))

            if do_checkpoint:
                ckpt = pl_module.on_save_checkpoint(ckpt)

            test_metrics = trainer.callback_metrics
            for key, value in test_metrics.items():
                if test_split_name == "test":
                    outputs[f"{test_split_name}_{key}"] = value
                else:
                    outputs[key] = value

        if do_checkpoint and config.get("eval.save_scores"):
            torch.save(ckpt, ckpt_path)

    outputs = {key: float(value) for key, value in outputs.items()}
    return outputs


def _setup_biencoder(config):
    # Dataset arguments
    data_folder = os.path.join(config.get("dataset.base"), config.get("dataset.name"))
    num_entities = config.get("dataset.num_entities")
    num_relations = config.get("dataset.num_relations")
    train_split_name = config.get("dataset.splits.train")
    valid_split_name = config.get("dataset.splits.valid")
    test_split_names = config.get("dataset.splits.test")
    if not isinstance(test_split_names, list):
        test_split_names = [test_split_names]

    # Model and training arguments
    batch_size = config.get("train.batch_size")
    lr = config.get("train.lr")
    use_bce_loss = config.get("train.use_bce_loss")
    use_margin_loss = config.get("train.use_margin_loss")
    use_relation_cls_loss = config.get("train.use_relation_cls_loss")
    relation_cls_loss_weight = config.get("train.relation_cls_loss_weight")
    margin = config.get("train.margin")
    warmup_frac = config.get("train.warmup_frac")
    num_neg_per_pos_train = config.get("train.negative_samples.num_neg_per_pos")

    # Model-specific arguments
    lm_model_name = config.get("lm.model_name")
    lm_pool_strategy = config.get("lm.pool_strategy")
    lm_max_length = config.get("lm.max_length")
    subj_repr = config.get("dataset.text.subj_repr")
    obj_repr = config.get("dataset.text.obj_repr")

    # Set up model and train/valid/test datasets
    pl_module = BiEncoder(
        num_entities=num_entities,
        num_relations=num_relations,
        lr=lr,
        use_bce_loss=use_bce_loss,
        use_margin_loss=use_margin_loss,
        use_relation_cls_loss=use_relation_cls_loss,
        relation_cls_loss_weight=relation_cls_loss_weight,
        margin=margin,
        batch_size=batch_size,
        warmup_frac=warmup_frac,
        model_name=lm_model_name,
        pool_strategy=lm_pool_strategy,
    )

    train_collate_fn = BiEncoderTrainDataset.collate_fn
    eval_collate_fn = BiEncoderEvalDataset.collate_fn

    train_datasets = {
        query_mode: BiEncoderTrainDataset(
            folder=data_folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=train_split_name,
            query_mode=query_mode,
            num_neg_per_pos=num_neg_per_pos_train,
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=lm_model_name,
            max_length=lm_max_length,
        )
        for query_mode in QUERY_MODES
    }
    valid_datasets = {
        query_mode: BiEncoderEvalDataset(
            folder=data_folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=valid_split_name,
            query_mode=query_mode,
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=lm_model_name,
            max_length=lm_max_length,
            negative_candidate_path=None,
        )
        for query_mode in QUERY_MODES
    }

    test_datasets_by_split = {}
    for test_split_name in test_split_names:
        test_datasets = {
            query_mode: BiEncoderEvalDataset(
                folder=data_folder,
                num_entities=num_entities,
                num_relations=num_relations,
                split_name=test_split_name,
                query_mode=query_mode,
                subj_repr=subj_repr,
                obj_repr=obj_repr,
                model_name=lm_model_name,
                max_length=lm_max_length,
                negative_candidate_path=None,
            )
            for query_mode in QUERY_MODES
        }
        test_datasets_by_split[test_split_name] = test_datasets

    outputs = {
        "pl_module": pl_module,
        "train_collate_fn": train_collate_fn,
        "eval_collate_fn": eval_collate_fn,
        "train_datasets": train_datasets,
        "valid_datasets": valid_datasets,
        "test_datasets_by_split": test_datasets_by_split,
    }
    return outputs


def _setup_crossencoder(config):
    # Dataset arguments
    data_folder = os.path.join(config.get("dataset.base"), config.get("dataset.name"))
    num_entities = config.get("dataset.num_entities")
    num_relations = config.get("dataset.num_relations")
    train_split_name = config.get("dataset.splits.train")
    valid_split_name = config.get("dataset.splits.valid")
    test_split_names = config.get("dataset.splits.test")
    if not isinstance(test_split_names, list):
        test_split_names = [test_split_names]

    # Model and training arguments
    batch_size = config.get("train.batch_size")
    lr = config.get("train.lr")
    use_bce_loss = config.get("train.use_bce_loss")
    use_margin_loss = config.get("train.use_margin_loss")
    use_relation_cls_loss = config.get("train.use_relation_cls_loss")
    relation_cls_loss_weight = config.get("train.relation_cls_loss_weight")
    margin = config.get("train.margin")
    warmup_frac = config.get("train.warmup_frac")
    num_neg_per_pos_train = config.get("train.negative_samples.num_neg_per_pos")

    # Model-specific arguments
    lm_model_name = config.get("lm.model_name")
    lm_pool_strategy = config.get("lm.pool_strategy")
    lm_max_length = config.get("lm.max_length")
    subj_repr = config.get("dataset.text.subj_repr")
    obj_repr = config.get("dataset.text.obj_repr")

    # Set up model and train/valid/test datasets
    pl_module = CrossEncoder(
        num_entities=num_entities,
        num_relations=num_relations,
        lr=lr,
        use_bce_loss=use_bce_loss,
        use_margin_loss=use_margin_loss,
        use_relation_cls_loss=use_relation_cls_loss,
        relation_cls_loss_weight=relation_cls_loss_weight,
        margin=margin,
        batch_size=batch_size,
        warmup_frac=warmup_frac,
        model_name=lm_model_name,
        pool_strategy=lm_pool_strategy,
    )
    train_collate_fn = CrossEncoderTrainDataset.collate_fn
    eval_collate_fn = CrossEncoderEvalDataset.collate_fn

    train_datasets = {
        query_mode: CrossEncoderTrainDataset(
            folder=data_folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=train_split_name,
            query_mode=query_mode,
            num_neg_per_pos=num_neg_per_pos_train,
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=lm_model_name,
            max_length=lm_max_length,
        )
        for query_mode in QUERY_MODES
    }

    valid_datasets = {
        query_mode: CrossEncoderEvalDataset(
            folder=data_folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=valid_split_name,
            query_mode=query_mode,
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=lm_model_name,
            max_length=lm_max_length,
            negative_candidate_path=None,
        )
        for query_mode in QUERY_MODES
    }

    test_datasets_by_split = {}
    for test_split_name in test_split_names:
        test_datasets = {
            query_mode: CrossEncoderEvalDataset(
                folder=data_folder,
                num_entities=num_entities,
                num_relations=num_relations,
                split_name=test_split_name,
                query_mode=query_mode,
                subj_repr=subj_repr,
                obj_repr=obj_repr,
                model_name=lm_model_name,
                max_length=lm_max_length,
                negative_candidate_path=None,
            )
            for query_mode in QUERY_MODES
        }

    test_datasets_by_split[test_split_name] = test_datasets

    outputs = {
        "pl_module": pl_module,
        "train_collate_fn": train_collate_fn,
        "eval_collate_fn": eval_collate_fn,
        "train_datasets": train_datasets,
        "valid_datasets": valid_datasets,
        "test_datasets_by_split": test_datasets_by_split,
    }
    return outputs


def _setup_crossencoder_reranker(config):
    # Dataset arguments
    data_folder = os.path.join(config.get("dataset.base"), config.get("dataset.name"))
    num_entities = config.get("dataset.num_entities")
    num_relations = config.get("dataset.num_relations")
    train_split_name = config.get("dataset.splits.train")
    valid_split_name = config.get("dataset.splits.valid")
    test_split_names = config.get("dataset.splits.test")
    if not isinstance(test_split_names, list):
        test_split_names = [test_split_names]

    # Model and training arguments
    batch_size = config.get("train.batch_size")
    lr = config.get("train.lr")
    use_bce_loss = config.get("train.use_bce_loss")
    use_margin_loss = config.get("train.use_margin_loss")
    use_relation_cls_loss = config.get("train.use_relation_cls_loss")
    relation_cls_loss_weight = config.get("train.relation_cls_loss_weight")
    margin = config.get("train.margin")
    warmup_frac = config.get("train.warmup_frac")
    num_neg_per_pos_train = config.get("train.negative_samples.num_neg_per_pos")

    # Model-specific arguments
    lm_model_name = config.get("lm.model_name")
    lm_pool_strategy = config.get("lm.pool_strategy")
    lm_max_length = config.get("lm.max_length")
    subj_repr = config.get("dataset.text.subj_repr")
    obj_repr = config.get("dataset.text.obj_repr")

    # Ensemble arguments
    base_ranker_ckpt_path = config.get("ensemble.base_ranker_checkpoint_path")
    answer_selector_ckpt_path = config.get("ensemble.answer_selector_checkpoint_path")
    answer_budget = config.get("ensemble.answer_budget")

    # Set up model and train/valid/test datasets
    pl_module = CrossEncoderReranker(
        num_entities=num_entities,
        num_relations=num_relations,
        lr=lr,
        use_bce_loss=use_bce_loss,
        use_margin_loss=use_margin_loss,
        use_relation_cls_loss=use_relation_cls_loss,
        relation_cls_loss_weight=relation_cls_loss_weight,
        margin=margin,
        batch_size=batch_size,
        warmup_frac=warmup_frac,
        model_name=lm_model_name,
        pool_strategy=lm_pool_strategy,
    )
    train_collate_fn = CrossEncoderTrainDataset.collate_fn
    eval_collate_fn = CrossEncoderRerankerEvalDataset.collate_fn

    train_datasets = {
        query_mode: CrossEncoderTrainDataset(
            folder=data_folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=train_split_name,
            query_mode=query_mode,
            num_neg_per_pos=num_neg_per_pos_train,
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=lm_model_name,
            max_length=lm_max_length,
        )
        for query_mode in QUERY_MODES
    }

    valid_datasets = {
        query_mode: CrossEncoderRerankerEvalDataset(
            folder=data_folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=valid_split_name,
            query_mode=query_mode,
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=lm_model_name,
            max_length=lm_max_length,
            base_ranker_ckpt_path=base_ranker_ckpt_path,
            answer_budget=answer_budget,
            answer_selector_ckpt_path=answer_selector_ckpt_path,
        )
        for query_mode in QUERY_MODES
    }

    test_datasets_by_split = {}
    for test_split_name in test_split_names:
        test_datasets = {
            query_mode: CrossEncoderRerankerEvalDataset(
                folder=data_folder,
                num_entities=num_entities,
                num_relations=num_relations,
                split_name=test_split_name,
                query_mode=query_mode,
                subj_repr=subj_repr,
                obj_repr=obj_repr,
                model_name=lm_model_name,
                max_length=lm_max_length,
                base_ranker_ckpt_path=base_ranker_ckpt_path,
                answer_budget=answer_budget,
                answer_selector_ckpt_path=answer_selector_ckpt_path,
            )
            for query_mode in QUERY_MODES
        }

        test_datasets_by_split[test_split_name] = test_datasets

    outputs = {
        "pl_module": pl_module,
        "train_collate_fn": train_collate_fn,
        "eval_collate_fn": eval_collate_fn,
        "train_datasets": train_datasets,
        "valid_datasets": valid_datasets,
        "test_datasets_by_split": test_datasets_by_split,
    }
    return outputs


def _setup_answer_selector(config):
    data_folder = os.path.join(config.get("dataset.base"), config.get("dataset.name"))
    num_entities = config.get("dataset.num_entities")
    num_relations = config.get("dataset.num_relations")
    train_split_name = config.get("dataset.splits.train")
    valid_split_name = config.get("dataset.splits.valid")
    test_split_names = config.get("dataset.splits.test")
    if not isinstance(test_split_names, list):
        test_split_names = [test_split_names]

    # Model and training arguments
    base_ranker_ckpt_path = config.get("answer_selector.base_ranker_checkpoint_path")
    hidden_channels = config.get("answer_selector.mlp.hidden_channels")
    dropout = config.get("answer_selector.mlp.dropout")
    quantiles = config.get("answer_selector.quantiles")

    # Set up model and train/valid/test datasets
    pl_module = AnswerSelector(
        num_entities=num_entities,
        num_relations=num_relations,
        lr=config.get("train.lr"),
        hidden_channels=hidden_channels,
        dropout=dropout,
        quantiles=quantiles,
    )

    train_collate_fn = AnswerSelectorDataset.collate_fn
    eval_collate_fn = AnswerSelectorDataset.collate_fn

    outputs = {
        "pl_module": pl_module,
        "train_collate_fn": train_collate_fn,
        "eval_collate_fn": eval_collate_fn,
    }

    if "train" in config.get("job-modes"):
        train_datasets = {
            query_mode: AnswerSelectorDataset(
                folder=data_folder,
                num_entities=num_entities,
                num_relations=num_relations,
                split_name=train_split_name,
                query_mode=query_mode,
                base_ranker_ckpt_path=base_ranker_ckpt_path,
                quantiles=quantiles,
            )
            for query_mode in QUERY_MODES
        }

        valid_datasets = {
            query_mode: AnswerSelectorDataset(
                folder=data_folder,
                num_entities=num_entities,
                num_relations=num_relations,
                split_name=valid_split_name,
                query_mode=query_mode,
                base_ranker_ckpt_path=base_ranker_ckpt_path,
                quantiles=quantiles,
            )
            for query_mode in QUERY_MODES
        }
        outputs["train_datasets"] = train_datasets
        outputs["valid_datasets"] = valid_datasets

    if "test" in config.get("job-modes"):
        test_datasets_by_split = {}
        for test_split_name in test_split_names:
            test_datasets = {
                query_mode: AnswerSelectorDataset(
                    folder=data_folder,
                    num_entities=num_entities,
                    num_relations=num_relations,
                    split_name=test_split_name,
                    query_mode=query_mode,
                    base_ranker_ckpt_path=base_ranker_ckpt_path,
                    quantiles=quantiles,
                )
                for query_mode in QUERY_MODES
            }
            test_datasets_by_split[test_split_name] = test_datasets
        outputs["test_datasets_by_split"] = test_datasets_by_split

    return outputs


def _setup_score_eval(config):
    data_folder = os.path.join(config.get("dataset.base"), config.get("dataset.name"))
    num_entities = config.get("dataset.num_entities")
    num_relations = config.get("dataset.num_relations")
    valid_split_name = config.get("dataset.splits.valid")
    test_split_names = config.get("dataset.splits.test")
    if not isinstance(test_split_names, list):
        test_split_names = [test_split_names]

    # Set up model and train/valid/test datasets
    pl_module = QueryScoreEvalModule(
        num_entities=num_entities, num_relations=num_relations
    )
    eval_collate_fn = ScoreEvalDataset.collate_fn

    if config.get("kge.checkpoint_path") is not None:
        ckpt_path = config.get("kge.checkpoint_path")
        convert_from_libkge = True
    else:
        ckpt_path = config.get("score.checkpoint_path")
        convert_from_libkge = False

    valid_datasets = {
        query_mode: ScoreEvalDataset(
            folder=data_folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=valid_split_name,
            query_mode=query_mode,
            ckpt_path=ckpt_path,
            convert_from_libkge=convert_from_libkge,
        )
        for query_mode in QUERY_MODES
    }

    test_datasets_by_split = {}
    for test_split_name in test_split_names:
        test_datasets = {
            query_mode: ScoreEvalDataset(
                folder=data_folder,
                num_entities=num_entities,
                num_relations=num_relations,
                split_name=test_split_name,
                query_mode=query_mode,
                ckpt_path=ckpt_path,
                convert_from_libkge=convert_from_libkge,
            )
            for query_mode in QUERY_MODES
        }

        test_datasets_by_split[test_split_name] = test_datasets

    outputs = {
        "pl_module": pl_module,
        "eval_collate_fn": eval_collate_fn,
        "valid_datasets": valid_datasets,
        "test_datasets_by_split": test_datasets_by_split,
    }
    return outputs


def _setup_reranking_pipeline(config):
    data_folder = os.path.join(config.get("dataset.base"), config.get("dataset.name"))
    num_entities = config.get("dataset.num_entities")
    num_relations = config.get("dataset.num_relations")
    valid_split_name = config.get("dataset.splits.valid")
    test_split_names = config.get("dataset.splits.test")
    if not isinstance(test_split_names, list):
        test_split_names = [test_split_names]

    # Set up model and train/valid/test datasets
    pl_module = QueryScoreEvalModule(
        num_entities=num_entities, num_relations=num_relations
    )
    eval_collate_fn = RerankingDataset.collate_fn
    base_ranker_ckpt_path = config.get("ensemble.base_ranker_checkpoint_path")
    reranker_ckpt_path = config.get("ensemble.reranker_checkpoint_path")

    base_ranker_scores_valid = {
        query_mode: load_scores(base_ranker_ckpt_path, valid_split_name, query_mode)
        for query_mode in QUERY_MODES
    }

    reranker_scores_valid = {
        query_mode: load_scores(reranker_ckpt_path, valid_split_name, query_mode)
        for query_mode in QUERY_MODES
    }

    base_ranker_scores_test = {}
    reranker_scores_test = {}

    for test_split_name in test_split_names:
        base_ranker_scores_test[test_split_name] = {
            query_mode: load_scores(base_ranker_ckpt_path, test_split_name, query_mode)
            for query_mode in QUERY_MODES
        }

        reranker_scores_test[test_split_name] = {
            query_mode: load_scores(reranker_ckpt_path, test_split_name, query_mode)
            for query_mode in QUERY_MODES
        }

    answer_selector_ckpt_path = config.get("ensemble.answer_selector_checkpoint_path")
    answer_selector_type = config.get("ensemble.answer_selector_type")
    quantile = config.get("ensemble.quantile")
    top_k = config.get("ensemble.top_k")

    reranker_weight_head_batch = config.get("ensemble.reranker_weight_head_batch")
    reranker_weight_tail_batch = config.get("ensemble.reranker_weight_tail_batch")

    reranker_weights = {
        "head-batch": reranker_weight_head_batch,
        "tail-batch": reranker_weight_tail_batch,
    }

    valid_datasets = {
        query_mode: RerankingDataset(
            folder=data_folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=valid_split_name,
            query_mode=query_mode,
            base_ranker_ckpt_path=base_ranker_ckpt_path,
            reranker_ckpt_path=reranker_ckpt_path,
            base_ranker_scores=base_ranker_scores_valid[query_mode],
            reranker_scores=reranker_scores_valid[query_mode],
            reranker_weight=reranker_weights[query_mode],
            answer_selector_ckpt_path=answer_selector_ckpt_path,
            answer_selector_type=answer_selector_type,
            top_k=top_k,
            quantile=quantile,
        )
        for query_mode in QUERY_MODES
    }

    test_datasets_by_split = {}
    for test_split_name in test_split_names:
        test_datasets = {
            query_mode: RerankingDataset(
                folder=data_folder,
                num_entities=num_entities,
                num_relations=num_relations,
                split_name=test_split_name,
                query_mode=query_mode,
                base_ranker_ckpt_path=base_ranker_ckpt_path,
                reranker_ckpt_path=reranker_ckpt_path,
                base_ranker_scores=base_ranker_scores_test[test_split_name][query_mode],
                reranker_scores=reranker_scores_test[test_split_name][query_mode],
                reranker_weight=reranker_weights[query_mode],
                answer_selector_ckpt_path=answer_selector_ckpt_path,
                answer_selector_type=answer_selector_type,
                top_k=top_k,
                quantile=quantile,
            )
            for query_mode in QUERY_MODES
        }

        test_datasets_by_split[test_split_name] = test_datasets

    outputs = {
        "pl_module": pl_module,
        "eval_collate_fn": eval_collate_fn,
        "valid_datasets": valid_datasets,
        "test_datasets_by_split": test_datasets_by_split,
    }
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config_path", help="Path to configuration file")
    parser.add_argument("--search", action="store_true", help="Perform Ax search")

    args = parser.parse_args()

    config = Config.create_from_yaml(args.config_path)

    if args.search:
        search_job = AxSearchJob.create(config, main)
        search_job.run()
    else:
        job_outputs = main(config)
        output_df = pd.DataFrame.from_dict(job_outputs, orient="index")
        output_df.reset_index(inplace=True)
        df = output_df.transpose()
        header = df.iloc[0].values
        df = df[1:]
        df.columns = header

        metric_names = ["mrr", "mr", "hits@1", "hits@3", "hits@10"]
        cols = []

        if "valid_mrr" in header:
            cols.extend(["valid_" + metric for metric in metric_names])
        elif "mrr" in header:
            cols.extend(metric_names)

        if "test_mrr" in header:
            cols.extend(["test_" + metric for metric in metric_names])

        values = df.iloc[0][cols].values
        print(",".join([col for col in cols]))
        print(",".join([f"{val:.5f}" for val in values]))
