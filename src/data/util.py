import os
import pandas as pd
import torch
import numpy as np

from collections import defaultdict

from kge.model import KgeModel
from kge.util.io import load_checkpoint

from data.tokenize import Tokenizer


class TextRetriever(object):
    """Based off of https://github.com/rahuln/lm-bio-kgc/"""

    def __init__(
        self,
        dataset_dir,
        entity_filename="entity_text.tsv",
        relation_filename="relation_text.tsv",
    ):
        # Load file with entity text
        entity_filename = os.path.join(dataset_dir, entity_filename)
        self.entity_file = pd.read_table(entity_filename, index_col=0, na_filter=False)
        self.num_entities = len(self.entity_file)

        # Map each entity type to an ID
        if "ent_type" in self.entity_file.columns:
            self.has_entity_types = True
            self.entity_types = self.entity_file["ent_type"].unique()
            type2id = {  # string type : int type ID
                entity_type: i for i, entity_type in enumerate(self.entity_types)
            }

            # Map each entity ID to entity type ID
            self.ent2type = {  # int entity ID : int type ID
                i: type2id[self.entity_file["ent_type"].iloc[i]]
                for i in range(self.num_entities)
            }

            # Map each entity type int ID to a list of entity IDs
            type2ent = defaultdict(list)
            for ent_id, ent_type in self.ent2type.items():
                type2ent[ent_type].append(ent_id)
            self.type2ent = dict(type2ent)
        else:
            self.has_entity_types = False

        # Load file with relation text
        relation_filename = os.path.join(dataset_dir, relation_filename)
        self.relation_file = pd.read_table(
            relation_filename, index_col=0, na_filter=False
        )

        self.num_relations = len(self.relation_file)

    def get_entity_type(self, ent):
        if not self.has_entity_types:
            raise ValueError("dataset does not contain entity types")
        return self.ent2type[ent]

    def get_entities_of_type(self, ent_type):
        if not self.has_entity_types:
            raise ValueError("dataset does not contain entity types")
        return self.type2ent[ent_type]

    def get_entity_text(self, ent, reprs):
        text = []
        for col in reprs:
            if col in self.entity_file.columns:
                text.append(self.entity_file.iloc[ent][col])
        return ": ".join(text)

    def get_relation_text(self, rel):
        return self.relation_file.iloc[rel]["name"]


class BiEncoder(object):
    """Utility class for bi-encoding tokenized text"""

    def __init__(
        self, subj_repr, obj_repr, model_name, max_length, tokenize_relations=True
    ):
        self.subj_repr = subj_repr
        self.obj_repr = obj_repr
        self.model_name = model_name
        self.max_length = max_length
        self.tokenize_relations = tokenize_relations

        if self.subj_repr != self.obj_repr:
            raise ValueError("biencoder requires same subj/obj answer representations")

        self.tokenizer = Tokenizer(self.model_name, self.max_length)

    def encode_query(self, text_retriever, subj, rel, obj, query_mode):
        subj_text = text_retriever.get_entity_text(ent=subj, reprs=self.subj_repr)
        obj_text = text_retriever.get_entity_text(ent=obj, reprs=self.obj_repr)

        if self.tokenize_relations:
            rel_text = text_retriever.get_relation_text(rel=rel)

            if query_mode == "head-batch":
                query_segments = [subj_text, rel_text]
            else:
                query_segments = [obj_text, rel_text]
        else:
            if query_mode == "head-batch":
                query_segments = [subj_text]
            else:
                query_segments = [obj_text]

        return self.tokenizer(
            segments=query_segments, use_relations=self.tokenize_relations
        )

    def encode_answer(self, text_retriever, candidate_id):
        candidate_text = text_retriever.get_entity_text(
            ent=candidate_id, reprs=self.subj_repr
        )
        answer_segments = [candidate_text]
        return self.tokenizer(segments=answer_segments)


class CrossEncoder(object):
    """Utility class for crossencoding tokenized text"""

    def __init__(
        self, subj_repr, obj_repr, model_name, max_length, tokenize_relations=True
    ):
        self.subj_repr = subj_repr
        self.obj_repr = obj_repr
        self.tokenize_relations = tokenize_relations
        self.tokenizer = Tokenizer(model_name=model_name, max_length=max_length)

    def encode(self, text_retriever, query_mode, subj, rel, obj, candidate_id):
        # Tokenize the representations of the candidate entities
        if query_mode == "head-batch":
            obj = candidate_id  # replace the true object with the candidate
        else:
            subj = candidate_id  # replace the true subject with the candidate

        subj_text = text_retriever.get_entity_text(ent=subj, reprs=self.subj_repr)
        obj_text = text_retriever.get_entity_text(ent=obj, reprs=self.obj_repr)

        if self.tokenize_relations:
            rel_text = text_retriever.get_relation_text(rel=rel)
            segments = [subj_text, rel_text, obj_text]
        else:
            segments = [subj_text, obj_text]

        candidate_triple = self.tokenizer(
            segments=segments, use_relations=self.tokenize_relations
        )
        return candidate_triple


def load_kge_scores(checkpoint_path, triples, query_mode):
    checkpoint = load_checkpoint(checkpoint_path)
    kge_model = KgeModel.create_from(checkpoint)

    if not isinstance(triples, torch.Tensor):
        triples = torch.tensor(triples, device="cpu")
    subj = triples[:, 0]
    rel = triples[:, 1]
    obj = triples[:, 2]

    if query_mode == "head-batch":
        kge_scores = kge_model.score_sp(subj, rel).detach()
    else:
        kge_scores = kge_model.score_po(rel, obj).detach()

    return kge_scores


def load_scores(checkpoint_path, split_name, query_mode, key="scores"):
    if torch.cuda.device_count() > 0:
        map_location = "cuda"
    else:
        map_location = "cpu"

    ckpt = torch.load(checkpoint_path, map_location=map_location)
    scores = ckpt[f"{split_name}_{key}"][query_mode]

    return torch.tensor(scores, dtype=torch.float)


def load_answer_mask(
    split_name,
    query_mode,
    base_ranker_scores,
    answer_selector_ckpt_path=None,
    answer_selector_type="static",
    top_k=None,
    quantile=0.95,
):
    if top_k is not None:
        _, topk = torch.topk(base_ranker_scores, top_k, dim=1)
        answer_mask = torch.zeros(base_ranker_scores.size(), dtype=torch.bool)
        for query_idx, indices in enumerate(topk):
            answer_mask[query_idx, indices] = True
    elif answer_selector_ckpt_path is not None:
        # Dynamically select top-ranked answers for each query
        ckpt = torch.load(
            answer_selector_ckpt_path,
            map_location="cuda" if torch.cuda.device_count() > 0 else "cpu",
        )
        qkey = f"quantile_{quantile}"

        if answer_selector_type == "static":
            k = ckpt["valid_y_static"][query_mode][qkey]  # static uses dev set
            _, topk = torch.topk(base_ranker_scores, k, dim=1)
            answer_mask = torch.zeros(base_ranker_scores.size(), dtype=torch.bool)
            for query_idx, indices in enumerate(topk):
                answer_mask[query_idx, indices] = True
        else:
            ks = ckpt[f"{split_name}_y_proba"][query_mode][qkey]
            answer_mask = torch.zeros(base_ranker_scores.size(), dtype=torch.bool)

            for query_idx, k in enumerate(ks):
                scores = base_ranker_scores[query_idx]
                _, indices = torch.topk(scores, k)
                answer_mask[query_idx, indices] = True
    else:
        # Select all answers for each query
        answer_mask = torch.ones(base_ranker_scores.size(), dtype=torch.bool)

    return answer_mask


def _filter_candidates(subj, rel, obj, triples, query_mode="head-batch"):
    if query_mode == "head-batch":  # filter out true objects
        filter_ids = triples[(triples[:, 0] == subj) & (triples[:, 1] == rel)][:, 2]
    else:  # filter out true subjects
        filter_ids = triples[(triples[:, 1] == rel) & (triples[:, 2] == obj)][:, 0]
    return filter_ids
