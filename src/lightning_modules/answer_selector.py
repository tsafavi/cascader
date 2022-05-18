import torch
import numpy as np

from collections import defaultdict

from lightning_modules.base_kgc_module import BaseKGCModule
from model.mlp import MLP
from ranking import evaluate_ranks


class AnswerSelector(BaseKGCModule):
    def __init__(
        self,
        num_entities,
        num_relations,
        lr,
        hidden_channels,
        dropout,
        quantiles,
        min_k=10,
    ):
        super().__init__(num_entities=num_entities, num_relations=num_relations)

        self.save_hyperparameters()

        self.model = MLP(
            self.num_entities,
            self.hparams.hidden_channels,
            len(quantiles),  # one output for each quantile
            self.hparams.dropout,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        loss = 0.0
        for query_mode, minibatch in batch.items():
            query_scores = self._score_queries(minibatch)
            targets = minibatch.targets
            loss += quantile_loss(query_scores, targets, self.hparams.quantiles)

        self.log("train_loss", loss)
        return loss

    def _setup_eval(self):
        self.y_true = defaultdict(list)
        self.y_proba = defaultdict(lambda: defaultdict(list))
        self.y_static = dict()
        return super()._setup_eval()

    def validation_step(self, batch, batch_idx):
        for query_mode, minibatch in batch.items():
            y_proba = self._score_queries(minibatch).cpu().numpy().astype(np.int64)
            y_proba = np.clip(y_proba, self.hparams.min_k, self.num_entities)
            y_true = minibatch.targets.flatten().cpu().numpy()

            self.y_true[query_mode].extend(y_true)

            for quantile_idx, quantile in enumerate(self.hparams.quantiles):
                qkey = f"quantile_{quantile}"
                self.y_proba[query_mode][qkey].append(y_proba[:, quantile_idx])

            query_idx = minibatch.query_idx
            self.eval_scores[query_mode][query_idx] = minibatch.features

    def _teardown_eval(self):
        mean_metrics = {}
        self.ranks = defaultdict(lambda: defaultdict(list))  # {metric: mode: ranks}
        dataloaders = self._get_eval_dataloaders()

        for query_mode, loader in dataloaders.items():
            dataset = loader.dataset
            scores = self.eval_scores[query_mode].clone()

            self.y_true[query_mode] = np.asarray(self.y_true[query_mode])
            self.y_static[query_mode] = dataset.static_K

            for quantile in self.hparams.quantiles:
                qkey = f"quantile_{quantile}"
                top_k = np.concatenate(self.y_proba[query_mode][qkey])
                self.y_proba[query_mode][qkey] = top_k

                filt_scores = torch.ones(scores.size(), device=scores.device)
                filt_scores *= float("-Inf")

                for query_idx, k in enumerate(top_k):
                    _, indices = torch.topk(scores[query_idx], k=k)
                    filt_scores[query_idx][indices] = scores[query_idx][indices]

                outputs = evaluate_ranks(dataset, filt_scores, device=self.device)

                for metric, value in outputs.items():
                    if metric == "mrr":
                        quantile_key = f"{metric}_{qkey}"
                        self.ranks[quantile_key][query_mode].extend(value)

            self.y_proba[query_mode] = dict(self.y_proba[query_mode])

            mean_metrics.update(
                {
                    metric: np.mean(
                        self.ranks[metric]["head-batch"]
                        + self.ranks[metric]["tail-batch"]
                    )
                    for metric in self.ranks.keys()
                }
            )

        for metric, mean_value in mean_metrics.items():
            self.log(metric, mean_value, prog_bar=True)

        return mean_metrics

    def on_save_checkpoint(self, checkpoint):
        checkpoint[f"{self.eval_split}_y_true"] = dict(self.y_true)
        checkpoint[f"{self.eval_split}_y_proba"] = dict(self.y_proba)
        checkpoint[f"{self.eval_split}_y_static"] = dict(self.y_static)
        return checkpoint

    def _score_queries(self, minibatch):
        input = torch.sort(minibatch.features, dim=1)[0]
        query_scores = self.model(input)
        return query_scores


def quantile_loss(preds, target, quantiles):
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss
