import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from collections import defaultdict

from ranking import evaluate_ranks


class BaseKGCModule(pl.LightningModule):
    def __init__(self, num_entities, num_relations):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.query_modes = ["head-batch", "tail-batch"]

    def setup(self, stage=None):
        self.stage = stage

    def training_step(self, batch, batch_idx):
        """Return training loss for this batch"""
        raise NotImplementedError

    def on_validation_epoch_start(self):
        self._setup_eval()
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        for query_mode, minibatch in batch.items():
            outputs = self._score_triples(minibatch=minibatch, query_mode=query_mode)
            triple_scores = outputs["triple_scores"].flatten()
            query_idx = minibatch.query_idx
            answer_idx = minibatch.answer_idx
            self.eval_scores[query_mode][query_idx, answer_idx] = triple_scores

    def on_validation_epoch_end(self):
        self._teardown_eval()
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        self._setup_eval()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self._teardown_eval()
        return super().on_test_epoch_end()

    def on_save_checkpoint(self, checkpoint):
        checkpoint[f"{self.eval_split}_ranks"] = self.ranks
        checkpoint[f"{self.eval_split}_scores"] = self.eval_scores
        return checkpoint

    def _get_eval_dataloaders(self):
        if self.stage == "fit" or self.stage == "validate":
            dataloaders = self.val_dataloader().loaders
        else:
            dataloaders = self.test_dataloader().loaders

        return dataloaders

    def _set_eval_split(self):
        dataloaders = self._get_eval_dataloaders()
        self.eval_split = list(dataloaders.values())[0].dataset.split_name

    def _setup_eval(self):
        """Set up the module for evaluation. Create separate empty score matrices
        of shape (num_queries,num_answers) for both subj/obj queries in this split,
        and pass back to the module."""
        dataloaders = self._get_eval_dataloaders()
        self._set_eval_split()
        self.eval_scores = {}  # {query-mode: query scores [n queries x n answers]}

        for query_mode, loader in dataloaders.items():
            dataset = loader.dataset

            self.eval_scores[query_mode] = torch.ones(
                [dataset.num_triples, dataset.num_neg_per_pos + 1],  # account for pos
                device=self.device,
                dtype=torch.float,
            )
            self.eval_scores[query_mode] *= float("-Inf")  # for safety in ranking

    def _teardown_eval(self):
        """Finish a round of evaluation for this module. Compute the ranks of the true
        answers for both subj/obj queries in this split given the ranking scores.
        Then, log the mean metrics and save the eval ranks of each example."""
        self.ranks = defaultdict(lambda: defaultdict(list))  # {metric: mode: ranks}
        dataloaders = self._get_eval_dataloaders()

        for query_mode, loader in dataloaders.items():
            dataset = loader.dataset
            scores = self.eval_scores[query_mode]
            ranks = evaluate_ranks(dataset, scores, device=self.device)
            for key, value in ranks.items():
                self.ranks[key][query_mode].extend(value)

        outputs = self._compute_ranks()
        return outputs

    def _compute_ranks(self):
        """Compute mean ranking metrics and save all evaluation statistics"""
        mean_metrics = {
            metric: np.mean(
                self.ranks[metric]["head-batch"] + self.ranks[metric]["tail-batch"],
                dtype=np.float64
            )
            for metric in self.ranks.keys()
        }

        for metric, mean_value in mean_metrics.items():
            self.log(metric, mean_value, prog_bar=True)

        # Convert evaluation scores/ranks to numpy arrays for checkpointing
        ranks_numpy = {}
        for metric in mean_metrics:
            ranks_numpy[metric] = {
                key: np.asarray(values) for key, values in self.ranks[metric].items()
            }
        self.ranks = ranks_numpy

        scores_numpy = {
            key: values.cpu().numpy() for key, values in self.eval_scores.items()
        }
        self.eval_scores = scores_numpy

        return mean_metrics

    def _score_triples(self, minibatch, query_mode):
        """
        :param minibatch: a batch of data
        :param query_mode: one of 'head-batch', 'tail-batch'
        :return outputs: a dict of outputs, with "triple_scores" as a key
        """
        raise NotImplementedError("base kgc module does not implement triple scoring")

    @staticmethod
    def loss(
        triple_scores,
        triple_targets,
        relation_scores=None,
        relation_targets=None,
        margin=1,
        use_bce_loss=True,
        use_margin_loss=True,
        use_relation_cls_loss=True,
        relation_cls_loss_weight=0.25,
    ):
        """Compute triple loss against binary targets and relation loss against
        multiclass targets"""
        loss = 0.0

        if not use_bce_loss and not use_margin_loss and not use_relation_cls_loss:
            raise ValueError("At least one type of loss must be specified")

        # First loss: Binary cross-entropy against binary triple targets
        triple_scores = triple_scores.view(triple_targets.shape)

        if use_bce_loss:
            bce_loss = F.binary_cross_entropy_with_logits(
                triple_scores, triple_targets.float()
            )
            loss += bce_loss

        # Second loss: Contrastive ranking loss against true/corrupted triples
        if use_margin_loss:
            pos_triple_scores = triple_scores[triple_targets == 1]
            npos = len(pos_triple_scores)
            neg_triple_scores = triple_scores[triple_targets == 0]
            nneg = len(neg_triple_scores)
            if npos and nneg:
                pos_triple_scores = pos_triple_scores.repeat_interleave(nneg).flatten()
                neg_triple_scores = neg_triple_scores.repeat(npos).flatten()
                margin_loss = F.margin_ranking_loss(
                    pos_triple_scores,
                    neg_triple_scores,
                    torch.ones_like(pos_triple_scores),
                    margin=margin,
                )
                loss += margin_loss

        # Final loss: Relation classification loss
        if (
            use_relation_cls_loss
            and relation_scores is not None
            and relation_targets is not None
        ):
            relation_cls_loss = F.cross_entropy(relation_scores, relation_targets)
            loss += relation_cls_loss_weight * relation_cls_loss

        return loss
