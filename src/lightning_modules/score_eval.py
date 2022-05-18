from lightning_modules.base_kgc_module import BaseKGCModule


class QueryScoreEvalModule(BaseKGCModule):
    def __init__(self, num_entities, num_relations):
        super().__init__(num_entities=num_entities, num_relations=num_relations)

    def validation_step(self, batch, batch_idx):
        for query_mode, minibatch in batch.items():
            triple_scores = minibatch.triple_scores
            query_idx = minibatch.query_idx
            self.eval_scores[query_mode][query_idx] = triple_scores

