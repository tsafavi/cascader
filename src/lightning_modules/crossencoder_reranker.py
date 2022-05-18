import torch
import numpy as np
from collections import defaultdict

from ranking import evaluate_ranks
from lightning_modules.crossencoder import CrossEncoder


class CrossEncoderReranker(CrossEncoder):
    def __init__(
        self,
        num_entities,
        num_relations,
        lr,
        use_bce_loss,
        use_margin_loss,
        use_relation_cls_loss,
        relation_cls_loss_weight,
        margin,
        batch_size,
        warmup_frac,
        model_name,
        pool_strategy,
    ):
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            lr=lr,
            use_bce_loss=use_bce_loss,
            use_margin_loss=use_margin_loss,
            use_relation_cls_loss=use_relation_cls_loss,
            relation_cls_loss_weight=relation_cls_loss_weight,
            margin=margin,
            batch_size=batch_size,
            warmup_frac=warmup_frac,
            model_name=model_name,
            pool_strategy=pool_strategy,
        )

    def _teardown_eval(self):
        dataloaders = self._get_eval_dataloaders()

        for query_mode, loader in dataloaders.items():
            dataset = loader.dataset

            # Save evaluated score subset to full |Q| x |A| score matrix
            partial_scores = self.eval_scores[query_mode]
            num_queries = len(dataset.query_mask)
            num_answers = dataset.num_entities
            full_scores = torch.zeros(
                (num_queries, num_answers), dtype=torch.float, device=self.device
            )

            query_mask = dataset.query_mask
            answer_mask = dataset.answer_mask

            for query_idx, (do_query, answer_idx, scores) in enumerate(
                zip(query_mask, answer_mask, partial_scores)
            ):
                if do_query:
                    full_scores[query_idx, answer_idx] = scores

            self.eval_scores[query_mode] = full_scores

        self.ranks = defaultdict(lambda: defaultdict(list))  # {metric: mode: ranks}
        dataloaders = self._get_eval_dataloaders()

        for query_mode, loader in dataloaders.items():
            dataset = loader.dataset
            scores = self.eval_scores[query_mode]

            partial_triples = dataset.triples.copy()
            dataset.triples = dataset.filter_splits[dataset.split_name]
            ranks = evaluate_ranks(dataset, scores, device=self.device)
            for key, value in ranks.items():
                self.ranks[key][query_mode].extend(value)

            dataset.triples = partial_triples

        outputs = self._compute_ranks()
        return outputs