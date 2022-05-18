import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from lightning_modules.base_kgc_module import BaseKGCModule


class BaseDenseRetrievalModule(BaseKGCModule):
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
    ):
        super().__init__(num_entities=num_entities, num_relations=num_relations)
        self.save_hyperparameters()

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

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=2e-5)

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
            outputs = self.model(
                input_ids_A=minibatch.query.input_ids,
                token_type_ids_A=minibatch.query.token_type_ids,
                attention_mask_A=minibatch.query.attention_mask,
                input_ids_B=minibatch.answer.input_ids,
                token_type_ids_B=minibatch.answer.token_type_ids,
                attention_mask_B=minibatch.answer.attention_mask,
            )

            triple_scores = outputs["triple_scores"]
            relation_scores = outputs["relation_scores"]

            loss += BaseKGCModule.loss(  # first loss: score-based
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

            if "dists" in outputs:
                dists = outputs["dists"]
                loss += BaseKGCModule.loss(  # optional second loss: distance-based
                    triple_scores=dists,
                    triple_targets=minibatch.targets,
                    margin=self.hparams.margin,
                    use_bce_loss=False,
                    use_margin_loss=True,
                    use_relation_cls_loss=False,
                )

        self.log("train_loss", loss)
        return loss

    def _setup_eval(self):
        super()._setup_eval()

        self._cache_queries()
        self._cache_answers()

    def _cache_queries(self):
        dataloaders = self._get_eval_dataloaders()
        self.query_embeddings = {}

        for query_mode, loader in dataloaders.items():
            query_dataset = loader.dataset.query_dataset
            query_loader = DataLoader(
                query_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=type(query_dataset).collate_fn
            )

            self.query_embeddings[query_mode] = []
            for minibatch in tqdm(query_loader, desc=f"Caching {query_mode} queries"):
                queries = minibatch.queries.to(self.device)
                query_embedding = self.model(
                    input_ids_A=queries.input_ids,
                    token_type_ids_A=queries.token_type_ids,
                    attention_mask_A=queries.attention_mask,
                )
                self.query_embeddings[query_mode].append(query_embedding)

            embeddings = self.query_embeddings[query_mode]
            self.query_embeddings[query_mode] = torch.vstack(embeddings)

    def _cache_answers(self):
        dataloaders = self._get_eval_dataloaders()
        dataset = dataloaders["head-batch"].dataset
        self.answer_embeddings = []

        answer_dataset = dataset.answer_dataset
        answer_loader = DataLoader(
            answer_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=type(answer_dataset).collate_fn
        )

        for minibatch in tqdm(answer_loader, desc="Caching answers"):
            answers = minibatch.answers.to(self.device)
            answer_embeddings = self.model(
                input_ids_A=answers.input_ids,
                token_type_ids_A=answers.token_type_ids,
                attention_mask_A=answers.attention_mask,
            )

            self.answer_embeddings.append(answer_embeddings)

        self.answer_embeddings = torch.vstack(self.answer_embeddings)

    def _score_triples(self, minibatch, query_mode):
        """Assumes queries/answers cached"""
        query_idx = minibatch.query_idx
        query_embeddings = self.query_embeddings[query_mode][query_idx]

        candidate_idx = minibatch.candidate_ids
        answer_embeddings = self.answer_embeddings[candidate_idx]

        outputs = self.model.score_pair(query_embeddings, answer_embeddings)
        return outputs
