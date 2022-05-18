from model.text_encoder import TextBiEncoder
from lightning_modules.base_kgc_module import BaseKGCModule
from lightning_modules.base_dense_retrieval import BaseDenseRetrievalModule


class BiEncoder(BaseDenseRetrievalModule):
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
        )

        self.save_hyperparameters()
        self.model = TextBiEncoder(
            num_entities=num_entities,
            num_relations=num_relations,
            model_name=model_name,
            pool_strategy=pool_strategy,
        )