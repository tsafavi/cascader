import torch

from data.triple_dataset import EvalTripleDataset
from data.util import load_scores, load_kge_scores


class ScoreEvalDataset(EvalTripleDataset):
    def __init__(
        self,
        folder,
        num_entities,
        num_relations,
        split_name,
        query_mode,
        ckpt_path,
        convert_from_libkge=False,
    ):
        super().__init__(
            folder=folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=split_name,
            query_mode=query_mode,
            negative_candidate_path=None,
        )

        if convert_from_libkge:
            self.scores = load_kge_scores(ckpt_path, self.triples, self.query_mode)
        else:
            self.scores = load_scores(ckpt_path, self.split_name, self.query_mode)

    def __len__(self):
        return self.num_triples

    def __getitem__(self, idx):
        triple_scores = self.scores[idx]
        outputs = {"query_idx": idx, "triple_scores": triple_scores}
        return outputs

    @staticmethod
    def collate_fn(data):
        return QueryScoreDataBatch.create(data)


class QueryScoreDataBatch(object):
    def __init__(self, query_idx, triple_scores):
        self.query_idx = query_idx
        self.triple_scores = triple_scores

    def to(self, device):
        self.query_idx = self.query_idx.to(device)
        self.triple_scores = self.triple_scores.to(device)
        return self

    @staticmethod
    def create(data):
        query_idx = torch.tensor([example["query_idx"] for example in data])
        triple_scores = torch.vstack([example["triple_scores"] for example in data])
        return QueryScoreDataBatch(query_idx=query_idx, triple_scores=triple_scores)
