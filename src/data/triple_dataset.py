import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class TripleDataset(Dataset):
    def __init__(self, folder, num_entities, num_relations, split_name, query_mode):
        """
        :param folder: directory of dataset files
        :param num_entities: number of entities in the KG
        :param num_relations: number of relations in the KG
        :param split_name: str name of the split to load
        :param query_mode: one of {head-batch,tail-batch}
        """
        self.folder = folder
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.split_name = split_name
        self.query_mode = query_mode

        # Load all positive examples for this split
        split_path = os.path.join(self.folder, self.split_name + ".del")
        self.triples = pd.read_csv(split_path, header=None, sep="\t").values
        self.num_triples = len(self.triples)

        self.num_neg_per_pos = 0

    def __len__(self):
        """Size of dataset = num queries * (num negative candidates per pos + 1 pos)"""
        return self.num_triples * (self.num_neg_per_pos + 1)

    def __getitem__(self, idx):
        """Get either the original triple at this index, or a corrupted negative"""
        target = idx < self.num_triples

        if target:
            outputs = self._get_pos_at_idx(idx)
        else:  # will only branch if self.num_neg_per_pos > 0
            outputs = self._get_neg_at_idx(idx)

        query_idx = outputs["query_idx"]
        subj, rel, obj = self.triples[query_idx]

        outputs["subj"] = subj
        outputs["rel"] = rel
        outputs["obj"] = obj
        outputs["target"] = target
        return outputs

    def _get_pos_at_idx(self, idx):
        """Get the index of the query + a positive (true) answer.
        :param idx: query index in [self.num_triples,
            self.num_triples * (self.num_neg_per_pos + 1))
        :return outputs: dictionary with keys:
            - query_idx: row of query in query-answer scoring matrix
            - answer_idx: column of true answer in query-answer scoring matrix
            - candidate_id: ID of true answer entity
        """
        raise NotImplementedError

    def _get_neg_at_idx(self, idx):
        """Get the index of the query + a candidate (negative) answer.
        :param idx: query index in [self.num_triples,
            self.num_triples * (self.num_neg_per_pos + 1))
        :return outputs: dictionary with keys:
            - query_idx: row of query in query-answer scoring matrix
            - answer_idx: column of candidate answer in query-answer scoring matrix
            - candidate_id: ID of candidate answer entity
        """
        raise NotImplementedError

    def _get_neg_query_idx(self, idx):
        """Translate index into range of [0, self.num_triples)"""
        query_idx = (idx - self.num_triples) // self.num_neg_per_pos
        return query_idx

    def _get_neg_answer_idx(self, idx):
        """Translate index into range of [0, self.num_neg_per_pos)"""
        answer_idx = (idx - self.num_triples) % self.num_neg_per_pos
        return answer_idx


class TrainTripleDataset(TripleDataset):
    """Training data: Positives + a fixed number of randomly sampled negatives"""

    def __init__(
        self,
        folder,
        num_entities,
        num_relations,
        split_name,
        query_mode,
        num_neg_per_pos,
    ):
        """
        :param num_neg_per_pos: number of negative examples per positive example
        """
        super().__init__(
            folder=folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=split_name,
            query_mode=query_mode,
        )

        self.num_neg_per_pos = num_neg_per_pos

    def _get_pos_at_idx(self, idx):
        query_idx = idx
        if self.query_mode == "head-batch":
            candidate_id = self.triples[query_idx][2]  # candidate = tail
        else:
            candidate_id = self.triples[query_idx][0]  # candidate = head
        answer_idx = candidate_id

        outputs = {
            "query_idx": query_idx,
            "answer_idx": answer_idx,
            "candidate_id": candidate_id,
        }
        return outputs

    def _get_neg_at_idx(self, idx):
        """Randomly sample a candidate negative for this query"""
        query_idx = self._get_neg_query_idx(idx)
        candidate_id = np.random.randint(0, self.num_entities)
        answer_idx = candidate_id

        outputs = {
            "query_idx": query_idx,
            "answer_idx": answer_idx,
            "candidate_id": candidate_id,
        }
        return outputs

    @staticmethod
    def collate_fn(data):
        return PosNegTripleDataBatch.create(data)


class EvalTripleDataset(TripleDataset):
    """Testing data: Positives + a fixed number of precomputed, ordered negatives"""

    def __init__(
        self,
        folder,
        num_entities,
        num_relations,
        split_name,
        query_mode,
        negative_candidate_path=None,
    ):
        """
        :param negative_candidate_path: path to pre-computed negative candidates
        """
        super().__init__(
            folder=folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=split_name,
            query_mode=query_mode,
        )

        # Load all triples for filtering in evaluation
        self.filter_splits = {
            split: pd.read_csv(
                os.path.join(self.folder, split + ".del"), header=None, sep="\t"
            ).values
            for split in ("train", "valid", "test")
        }

        # Load or construct negative candidates
        if isinstance(negative_candidate_path, str):
            neg_path = os.path.join(self.folder, negative_candidate_path)
        else:
            neg_path = None

        if neg_path is None or not os.path.isfile(neg_path):  # use all entities as negs
            self.negative_ids = None
            self.num_neg_per_pos = self.num_entities - 1
            self.do_filter = True  # flag for filtering in evaluation
        else:  # for bio KGs only
            neg_ids = torch.load(neg_path, map_location="cpu")
            self.negative_ids = np.asarray(neg_ids[self.query_mode])
            self.num_neg_per_pos = self.negative_ids.shape[1]
            self.do_filter = False

    def _get_pos_at_idx(self, idx):
        """Get the index of the query + positive answer.
        :param idx: query index in [0, self.num_triples)
        :return outputs: dictionary with keys:
            - query_idx: row of query in query-answer scoring matrix
            - answer_idx: column of correct answer in query-answer scoring matrix (0)
            - candidate_id: ID of correct answer entity"""
        query_idx = idx
        if self.query_mode == "head-batch":
            candidate_id = self.triples[query_idx][2]  # candidate = tail
        else:
            candidate_id = self.triples[query_idx][0]  # candidate = head

        if self.negative_ids is not None:
            answer_idx = 0  # positives always come before negatives in scoring matrix
        else:
            answer_idx = candidate_id

        outputs = {
            "query_idx": query_idx,
            "answer_idx": answer_idx,
            "candidate_id": candidate_id,
        }
        return outputs

    def _get_neg_at_idx(self, idx):
        """Index into the negative candidates matrix to retrieve candidate entity"""
        query_idx = self._get_neg_query_idx(idx)
        neg_answer_idx = self._get_neg_answer_idx(idx)

        if self.negative_ids is not None:
            candidate_id = self.negative_ids[query_idx, neg_answer_idx]
            answer_idx = neg_answer_idx + 1  # add one because positives come first
        else:
            candidate_id = neg_answer_idx
            answer_idx = neg_answer_idx

        outputs = {
            "query_idx": query_idx,
            "candidate_id": candidate_id,
            "answer_idx": answer_idx,
        }
        return outputs

    @staticmethod
    def collate_fn(data):
        return PosNegTripleDataBatch.create(data)


class TripleDataBatch(object):
    """Batch of positive subj/rel/obj IDs"""

    def __init__(self, query_idx, answer_idx, subj_ids, rel_ids, obj_ids):
        self.query_idx = query_idx
        self.answer_idx = answer_idx
        self.subj_ids = subj_ids
        self.rel_ids = rel_ids
        self.obj_ids = obj_ids

    def to(self, device):
        self.query_idx = self.query_idx.to(device)
        self.answer_idx = self.answer_idx.to(device)
        self.subj_ids = self.subj_ids.to(device)
        self.rel_ids = self.rel_ids.to(device)
        self.obj_ids = self.obj_ids.to(device)
        return self

    @staticmethod
    def create(data):
        # Collate the batch indices and head/relation/tail IDs
        query_idx = torch.tensor(
            [example["query_idx"] for example in data], dtype=torch.long
        )
        answer_idx = torch.tensor(
            [example["answer_idx"] for example in data], dtype=torch.long
        )
        subj_ids = torch.tensor([example["subj"] for example in data], dtype=torch.long)
        rel_ids = torch.tensor([example["rel"] for example in data], dtype=torch.long)
        obj_ids = torch.tensor([example["obj"] for example in data], dtype=torch.long)

        return TripleDataBatch(
            query_idx=query_idx,
            answer_idx=answer_idx,
            subj_ids=subj_ids,
            rel_ids=rel_ids,
            obj_ids=obj_ids,
        )


class PosNegTripleDataBatch(TripleDataBatch):
    """Training batch with pos/neg entity IDs and labels"""

    def __init__(
        self, query_idx, answer_idx, subj_ids, rel_ids, obj_ids, candidate_ids, targets,
    ):
        super().__init__(query_idx, answer_idx, subj_ids, rel_ids, obj_ids)

        self.candidate_ids = candidate_ids
        self.targets = targets

    def to(self, device):
        super().to(device)
        self.candidate_ids = self.candidate_ids.to(device)
        self.targets = self.targets.to(device)
        return self

    @staticmethod
    def create(data):
        batch = TripleDataBatch.create(data)

        candidate_ids = torch.tensor(
            [example["candidate_id"] for example in data], dtype=torch.long
        )
        targets = torch.tensor(
            [example["target"] for example in data], dtype=torch.long
        )

        return PosNegTripleDataBatch(
            query_idx=batch.query_idx,
            answer_idx=batch.answer_idx,
            subj_ids=batch.subj_ids,
            rel_ids=batch.rel_ids,
            obj_ids=batch.obj_ids,
            candidate_ids=candidate_ids,
            targets=targets,
        )
