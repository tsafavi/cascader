import torch
import torch.nn.functional as F

from data.triple_dataset import EvalTripleDataset
from data.score_eval_dataset import QueryScoreDataBatch
from data.util import load_scores, load_answer_mask


class RerankingDataset(EvalTripleDataset):
    def __init__(
        self,
        folder,
        num_entities,
        num_relations,
        split_name,
        query_mode,
        base_ranker_ckpt_path,
        reranker_ckpt_path,
        base_ranker_scores=None,
        reranker_scores=None,
        reranker_weight=0.5,
        answer_selector_ckpt_path=None,
        answer_selector_type="static",
        top_k=None,
        quantile=0.95,
    ):
        super().__init__(
            folder=folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=split_name,
            query_mode=query_mode,
            negative_candidate_path=None,
        )

        if base_ranker_scores is None:
            self.base_ranker_scores = load_scores(
                base_ranker_ckpt_path, self.split_name, self.query_mode
            )
        else:
            self.base_ranker_scores = base_ranker_scores

        if reranker_scores is None:
            self.reranker_scores = load_scores(
                reranker_ckpt_path, self.split_name, self.query_mode
            )
        else:
            self.reranker_scores = reranker_scores

        self.reranker_weight = reranker_weight
        self.base_ranker_weight = 1 - self.reranker_weight

        self.answer_mask = load_answer_mask(
            self.split_name,
            self.query_mode,
            self.base_ranker_scores,
            answer_selector_ckpt_path=answer_selector_ckpt_path,
            answer_selector_type=answer_selector_type,
            top_k=top_k,
            quantile=quantile,
        )

    def __len__(self):
        return self.num_triples

    def __getitem__(self, idx):
        base_ranker_scores = self.base_ranker_scores[idx]
        answer_idx = self.answer_mask[idx]
        reranker_scores = self.reranker_scores[idx]
        reranked_scores = (
            self.base_ranker_weight * base_ranker_scores[answer_idx]
            + (1 - self.base_ranker_weight) * reranker_scores[answer_idx]
        )
        base_ranker_scores[answer_idx] = reranked_scores

        triple_scores = base_ranker_scores
        outputs = {"query_idx": idx, "triple_scores": triple_scores}
        return outputs

    @staticmethod
    def collate_fn(data):
        return QueryScoreDataBatch.create(data)
