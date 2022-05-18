import torch

from data.crossencoder_dataset import CrossEncoderEvalDataset
from data.util import load_scores, load_answer_mask


class CrossEncoderRerankerEvalDataset(CrossEncoderEvalDataset):
    def __init__(
        self,
        folder,
        num_entities,
        num_relations,
        split_name,
        query_mode,
        subj_repr,
        obj_repr,
        model_name,
        max_length,
        base_ranker_ckpt_path,
        top_k=None,
        tokenize_relations=True,
        negative_candidate_path=None,
    ):
        super().__init__(
            folder=folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=split_name,
            query_mode=query_mode,
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=model_name,
            max_length=max_length,
            tokenize_relations=tokenize_relations,
            negative_candidate_path=negative_candidate_path,
        )

        self.base_ranker_scores = load_scores(
            base_ranker_ckpt_path, self.split_name, self.query_mode
        )
        self.answer_mask = load_answer_mask(
            self.split_name,
            self.query_mode,
            self.base_ranker_scores,
            answer_selector_ckpt_path=None,
            top_k=top_k,
        )

        nonzero = torch.nonzero(self.answer_mask, as_tuple=False)
        nonzero = nonzero[:, 1].reshape(-1, nonzero.shape[0] // self.num_triples)
        self.answer_ids = nonzero.cpu().numpy()
        self.num_neg_per_pos = self.answer_ids.shape[1] - 1

    def __getitem__(self, idx):
        """Get either the original triple at this index, or a corrupted negative"""
        query_idx = idx // (self.num_neg_per_pos + 1)
        answer_idx = idx % (self.num_neg_per_pos + 1)
        candidate_id = self.answer_ids[query_idx, answer_idx]
        subj, rel, obj = self.triples[query_idx]

        candidate_triple = self.crossencoder.encode(
            text_retriever=self.text_retriever,
            query_mode=self.query_mode,
            subj=subj,
            rel=rel,
            obj=obj,
            candidate_id=candidate_id,
        )

        outputs = {
            "query_idx": query_idx,
            "answer_idx": answer_idx,
            "candidate_id": candidate_id,
            "subj": subj,
            "rel": rel,
            "obj": obj,
            "target": 0,
            "candidate_triple": candidate_triple,
        }

        return outputs
