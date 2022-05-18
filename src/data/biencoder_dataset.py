import torch

from torch.utils.data import Dataset

from data.tokenize import TokenizedDataBatch
from data.util import TextRetriever, BiEncoder
from data.triple_dataset import (
    TrainTripleDataset,
    EvalTripleDataset,
    PosNegTripleDataBatch,
)


class BiEncoderTrainDataset(TrainTripleDataset):
    def __init__(
        self,
        folder,
        num_entities,
        num_relations,
        split_name,
        query_mode,
        num_neg_per_pos,
        subj_repr,
        obj_repr,
        model_name,
        max_length,
        tokenize_relations=True,
    ):
        super().__init__(
            folder=folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=split_name,
            query_mode=query_mode,
            num_neg_per_pos=num_neg_per_pos,
        )

        self.text_retriever = TextRetriever(self.folder)

        self.biencoder = BiEncoder(
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=model_name,
            max_length=max_length,
            tokenize_relations=tokenize_relations,
        )

    def __getitem__(self, idx):
        """Return triple + separately encoded query/answer"""
        outputs = super().__getitem__(idx)
        outputs["query"] = self.biencoder.encode_query(
            text_retriever=self.text_retriever,
            subj=outputs["subj"],
            rel=outputs["rel"],
            obj=outputs["obj"],
            query_mode=self.query_mode,
        )
        outputs["answer"] = self.biencoder.encode_answer(
            text_retriever=self.text_retriever, candidate_id=outputs["candidate_id"]
        )
        return outputs

    @staticmethod
    def collate_fn(data):
        return BiEncoderTrainDataBatch.create(data)


class BiEncoderEvalDataset(EvalTripleDataset):
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
        tokenize_relations=True,
        negative_candidate_path=None,
    ):
        super().__init__(
            folder=folder,
            num_entities=num_entities,
            num_relations=num_relations,
            split_name=split_name,
            query_mode=query_mode,
            negative_candidate_path=negative_candidate_path,
        )

        self.query_dataset = BiEncoderQueryDataset(
            folder=self.folder,
            query_mode=self.query_mode,
            triples=self.triples,
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=model_name,
            max_length=max_length,
            tokenize_relations=tokenize_relations,
        )

        self.answer_dataset = BiEncoderAnswerDataset(
            folder=self.folder,
            num_entities=self.num_entities,
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=model_name,
            max_length=max_length,
            tokenize_relations=tokenize_relations
        )


class BiEncoderQueryDataset(Dataset):
    def __init__(
        self,
        folder,
        query_mode,
        triples,
        subj_repr,
        obj_repr,
        model_name,
        max_length,
        tokenize_relations=True,
    ):
        self.triples = triples
        self.num_triples = len(self.triples)

        self.query_mode = query_mode

        self.text_retriever = TextRetriever(folder)

        self.biencoder = BiEncoder(
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=model_name,
            max_length=max_length,
            tokenize_relations=tokenize_relations,
        )

    def __len__(self):
        return self.num_triples

    def __getitem__(self, idx):
        query_idx = idx
        subj, rel, obj = self.triples[query_idx]

        outputs = {}
        outputs["query_idx"] = query_idx
        outputs["query"] = self.biencoder.encode_query(
            text_retriever=self.text_retriever,
            subj=subj,
            rel=rel,
            obj=obj,
            query_mode=self.query_mode,
        )
        return outputs

    @staticmethod
    def collate_fn(data):
        return QueryDataBatch.create(data)


class BiEncoderAnswerDataset(Dataset):
    """Dataset of entities and their descriptions"""

    def __init__(
        self,
        folder,
        num_entities,
        subj_repr,
        obj_repr,
        model_name,
        max_length,
        tokenize_relations=True,
    ):
        self.text_retriever = TextRetriever(folder)
        self.num_entities = num_entities

        self.biencoder = BiEncoder(
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=model_name,
            max_length=max_length,
            tokenize_relations=tokenize_relations,
        )

    def __len__(self):
        return self.num_entities

    def __getitem__(self, idx):
        answer = self.biencoder.encode_answer(self.text_retriever, idx)
        outputs = {"answer_idx": idx, "answer": answer}
        return outputs

    @staticmethod
    def collate_fn(data):
        return AnswerDataBatch.create(data)


class BiEncoderTrainDataBatch(PosNegTripleDataBatch):
    def __init__(
        self,
        query_idx,
        answer_idx,
        subj_ids,
        rel_ids,
        obj_ids,
        candidate_ids,
        targets,
        query,
        answer,
    ):
        super().__init__(
            query_idx=query_idx,
            answer_idx=answer_idx,
            subj_ids=subj_ids,
            rel_ids=rel_ids,
            obj_ids=obj_ids,
            candidate_ids=candidate_ids,
            targets=targets,
        )

        self.query = query
        self.answer = answer

    def to(self, device):
        super().to(device)
        self.query = self.query.to(device)
        self.answer = self.answer.to(device)
        return self

    @staticmethod
    def create(data):
        batch = PosNegTripleDataBatch.create(data)

        query = TokenizedDataBatch.create(data, key="query")
        answer = TokenizedDataBatch.create(data, key="answer")

        return BiEncoderTrainDataBatch(
            query_idx=batch.query_idx,
            answer_idx=batch.answer_idx,
            subj_ids=batch.subj_ids,
            rel_ids=batch.rel_ids,
            obj_ids=batch.obj_ids,
            candidate_ids=batch.candidate_ids,
            targets=batch.targets,
            query=query,
            answer=answer,
        )


class QueryDataBatch(object):
    """Batch query IDs + tokenized queries"""

    def __init__(self, query_idx, queries):
        self.query_idx = query_idx
        self.queries = queries

    def to(self, device):
        self.query_idx = self.query_idx.to(device)
        self.queries = self.queries.to(device)
        return self

    @staticmethod
    def create(data):
        query_idx = torch.tensor(
            [example["query_idx"] for example in data], dtype=torch.long
        )
        queries = TokenizedDataBatch.create(data, key="query")
        return QueryDataBatch(query_idx, queries)


class AnswerDataBatch(object):
    """Batch answer (entity) IDs + tokenized textual descriptions"""

    def __init__(self, answer_idx, answers):
        self.answer_idx = answer_idx
        self.answers = answers

    def to(self, device):
        self.answer_idx = self.answer_idx.to(device)
        self.answers = self.answers.to(device)
        return self

    @staticmethod
    def create(data):
        answer_idx = torch.tensor(
            [example["answer_idx"] for example in data], dtype=torch.long
        )
        answers = TokenizedDataBatch.create(data, key="answer")
        return AnswerDataBatch(answer_idx, answers)
