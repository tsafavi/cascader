from data.triple_dataset import (
    TrainTripleDataset,
    EvalTripleDataset,
    PosNegTripleDataBatch,
)
from data.tokenize import TokenizedDataBatch
from data.util import TextRetriever, CrossEncoder


class CrossEncoderTrainDataset(TrainTripleDataset):
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

        self.crossencoder = CrossEncoder(
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=model_name,
            max_length=max_length,
            tokenize_relations=tokenize_relations,
        )

    def __getitem__(self, idx):
        outputs = super().__getitem__(idx)
        candidate_triple = self.crossencoder.encode(
            text_retriever=self.text_retriever,
            query_mode=self.query_mode,
            subj=outputs["subj"],
            rel=outputs["rel"],
            obj=outputs["obj"],
            candidate_id=outputs["candidate_id"],
        )
        outputs["candidate_triple"] = candidate_triple
        return outputs

    @staticmethod
    def collate_fn(data):
        return CrossEncoderDataBatch.create(data)


class CrossEncoderEvalDataset(EvalTripleDataset):
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

        self.text_retriever = TextRetriever(self.folder)

        self.crossencoder = CrossEncoder(
            subj_repr=subj_repr,
            obj_repr=obj_repr,
            model_name=model_name,
            max_length=max_length,
            tokenize_relations=tokenize_relations,
        )

    def __getitem__(self, idx):
        outputs = super().__getitem__(idx)
        candidate_triple = self.crossencoder.encode(
            text_retriever=self.text_retriever,
            query_mode=self.query_mode,
            subj=outputs["subj"],
            rel=outputs["rel"],
            obj=outputs["obj"],
            candidate_id=outputs["candidate_id"],
        )
        outputs["candidate_triple"] = candidate_triple
        return outputs

    @staticmethod
    def collate_fn(data):
        return CrossEncoderDataBatch.create(data)


class CrossEncoderDataBatch(PosNegTripleDataBatch):
    def __init__(
        self,
        query_idx,
        answer_idx,
        subj_ids,
        rel_ids,
        obj_ids,
        candidate_ids,
        targets,
        candidate_triples,
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

        self.candidate_triples = candidate_triples

    def to(self, device):
        super().to(device)
        self.candidate_triples = self.candidate_triples.to(device)
        return self

    @staticmethod
    def create(data):
        batch = PosNegTripleDataBatch.create(data)
        candidate_triples = TokenizedDataBatch.create(data, key="candidate_triple")

        return CrossEncoderDataBatch(
            query_idx=batch.query_idx,
            answer_idx=batch.answer_idx,
            subj_ids=batch.subj_ids,
            rel_ids=batch.rel_ids,
            obj_ids=batch.obj_ids,
            candidate_ids=batch.candidate_ids,
            targets=batch.targets,
            candidate_triples=candidate_triples,
        )
