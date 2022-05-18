from transformers import AdamW, get_linear_schedule_with_warmup

from model.text_encoder import TextScorer
from lightning_modules.base_kgc_module import BaseKGCModule


class CrossEncoder(BaseKGCModule):
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
        super().__init__(num_entities=num_entities, num_relations=num_relations)
        self.save_hyperparameters()

        self.model = TextScorer(
            num_entities=num_entities,
            num_relations=num_relations,
            model_name=self.hparams.model_name,
            pool_strategy=self.hparams.pool_strategy,
        )

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr, eps=2e-5
        )

        # Calculate total and warmup steps
        query_modes = list(self.train_dataloader().keys())
        datasets = [
            self.train_dataloader()[query_mode].dataset for query_mode in query_modes
        ]

        self.total_steps = (
            sum([len(dataset) for dataset in datasets]) // self.hparams.batch_size
        )
        self.warmup_steps = int(self.hparams.warmup_frac * self.total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = 0.0
        for query_mode, minibatch in batch.items():
            outputs = self._score_triples(minibatch=minibatch, query_mode=query_mode)
            triple_scores = outputs["triple_scores"]
            relation_scores = outputs["relation_scores"]
            loss += BaseKGCModule.loss(
                triple_scores=triple_scores,
                triple_targets=minibatch.targets,
                relation_scores=relation_scores,
                relation_targets=minibatch.rel_ids,
                margin=self.hparams.margin,
                use_bce_loss=self.hparams.use_bce_loss,
                use_margin_loss=self.hparams.use_margin_loss,
                use_relation_cls_loss=self.hparams.use_relation_cls_loss,
                relation_cls_loss_weight=self.hparams.relation_cls_loss_weight,
            )

        self.log("train_loss", loss)
        return loss

    def _score_triples(self, minibatch, query_mode):
        triples = minibatch.candidate_triples
        return self.model(
            input_ids=triples.input_ids,
            token_type_ids=triples.token_type_ids,
            attention_mask=triples.attention_mask,
        )