import torch
import numpy as np

from data.triple_dataset import EvalTripleDataset
from data.util import load_scores
from ranking import evaluate_ranks


class AnswerSelectorDataset(EvalTripleDataset):
    def __init__(
        self,
        folder,
        num_entities,
        num_relations,
        split_name,
        query_mode,
        base_ranker_ckpt_path,
        quantiles,
        min_K=10,
    ):
        if split_name in ("valid_train", "valid_dev"):
            true_split_name = "valid"
            filter_splits = ["train"]
        else:
            true_split_name = split_name
            filter_splits = ["train", "valid"]

        super().__init__(
            folder=folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=true_split_name,
            query_mode=query_mode,
        )

        model_scores = load_scores(
            base_ranker_ckpt_path, self.split_name, self.query_mode
        )

        ranks = evaluate_ranks(
            self,
            model_scores,
            device=model_scores.device,
            filter_splits=filter_splits,
            metrics=["mr"],
            return_filtered_scores=True,
            replace_na=True,
        )

        self.features = ranks["filtered_scores"]
        self.targets = ranks["mr"].astype(np.int64)

        static_K = np.clip(np.quantile(self.targets, quantiles), min_K, num_entities)
        self.static_K = {
            f"quantile_{quantile}": int(K) for quantile, K in zip(quantiles, static_K)
        }

        # Split valid into train/test
        if split_name in ("valid_train", "valid_dev"):
            N = self.num_triples
            N_train = N // 2
            mask = np.zeros(N, dtype=bool)

            if split_name == "valid_train":
                mask[:N_train] = True
            else:
                mask[N_train:] = True

            self.features = self.features[mask]
            self.targets = self.targets[mask]
            self.triples = self.triples[mask]
            self.num_triples = len(self.triples)

    def __len__(self):
        return self.num_triples

    def __getitem__(self, idx):
        outputs = {
            "query_idx": idx,
            "target": self.targets[idx],
            "features": self.features[idx],
        }
        return outputs

    @staticmethod
    def collate_fn(data):
        return SelectorDataBatch.create(data)


class SelectorDataBatch(object):
    def __init__(self, query_idx, features, targets):
        self.query_idx = query_idx
        self.features = features
        self.targets = targets

    def to(self, device):
        self.query_idx = self.query_idx.to(device)
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        return self

    @staticmethod
    def create(data):
        query_idx = torch.tensor([example["query_idx"] for example in data])
        features = torch.vstack([example["features"] for example in data])
        targets = torch.tensor([example["target"] for example in data]).long()
        return SelectorDataBatch(
            query_idx=query_idx, features=features, targets=targets
        )
