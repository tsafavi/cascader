import torch

from transformers import AutoModel


class TextEncoder(torch.nn.Module):
    """Encode tokenized text with a Transformer LM"""

    def __init__(self, num_entities, num_relations, model_name, pool_strategy):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations

        # initialize transformer encoder
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.pool_strategy = pool_strategy

    def forward(self, input_ids, token_type_ids, attention_mask):
        # pass input through encoder
        encoder_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )

        # pool embeddings
        outputs = _pool_lm(
            encoder_output, attention_mask, pool_strategy=self.pool_strategy
        )
        return outputs

    def reset_parameters(self):
        self.encoder = AutoModel.from_pretrained(self.model_name)


class TextBiEncoder(TextEncoder):
    """Encode tokenized text in separate query/answer segments"""

    def __init__(self, num_entities, num_relations, model_name, pool_strategy):
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            model_name=model_name,
            pool_strategy=pool_strategy,
        )

        self.triple_score_head = torch.nn.Linear(self.hidden_size * 4, 1)
        self.relation_cls_head = torch.nn.Linear(
            self.hidden_size * 4, self.num_relations
        )

    def forward(
        self,
        input_ids_A,
        token_type_ids_A,
        attention_mask_A,
        input_ids_B=None,
        token_type_ids_B=None,
        attention_mask_B=None,
    ):
        embeddings_A = super().forward(input_ids_A, token_type_ids_A, attention_mask_A)

        if input_ids_B is not None:  # encode A and B, then score pair
            embeddings_B = super().forward(
                input_ids_B, token_type_ids_B, attention_mask_B
            )
            return self.score_pair(embeddings_A, embeddings_B)

        # Encode A only without scoring
        return embeddings_A

    def score_pair(self, query_embeddings, answer_embeddings):
        """Interactive query/answer embedding concatenation"""
        concat_embeddings = torch.cat(
            [
                query_embeddings,
                torch.mul(query_embeddings, answer_embeddings),
                torch.subtract(query_embeddings, answer_embeddings),
                answer_embeddings,
            ],
            dim=1,
        )

        triple_score_head = self.triple_score_head.to(concat_embeddings.device)
        triple_scores = triple_score_head(concat_embeddings)

        relation_cls_head = self.relation_cls_head.to(concat_embeddings.device)
        relation_scores = relation_cls_head(concat_embeddings)

        dists = torch.sqrt(
            torch.sum(torch.pow(query_embeddings - answer_embeddings, 2), dim=1)
        )

        outputs = {
            "triple_scores": triple_scores,
            "relation_scores": relation_scores,
            "dists": dists,
        }
        return outputs


class LateInteractionEncoder(TextEncoder):
    """Encode tokenized text in separate query/answer segments, and
    compute a late interaction mechanism over the output token embeddings"""

    def __init__(self, num_entities, num_relations, model_name, embed_dim):
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            model_name=model_name,
            pool_strategy=None,
        )

        self.embed_project_head = torch.nn.Linear(self.hidden_size, embed_dim)
        self.relation_cls_head = torch.nn.Linear(embed_dim * 4, self.num_relations)

    def forward(
        self,
        input_ids_A,
        token_type_ids_A,
        attention_mask_A,
        input_ids_B=None,
        token_type_ids_B=None,
        attention_mask_B=None,
    ):
        token_embed_A = super().forward(input_ids_A, token_type_ids_A, attention_mask_A)
        token_embed_A = self.embed_project_head(token_embed_A)
        token_embed_A = torch.nn.functional.normalize(token_embed_A, p=2, dim=2)

        if input_ids_B is not None:
            token_embed_B = super().forward(
                input_ids_B, token_type_ids_B, attention_mask_B
            )
            token_embed_B = self.embed_project_head(token_embed_B)
            token_embed_B = torch.nn.functional.normalize(token_embed_B, p=2, dim=2)

            return self.score_pair(
                token_embed_A=token_embed_A, token_embed_B=token_embed_B,
            )

        return token_embed_A

    def score_pair(self, token_embed_A, token_embed_B):
        triple_scores = (token_embed_A @ token_embed_B.permute(0, 2, 1)).max(2).values.sum(1)

        mean_embed_A = torch.mean(token_embed_A, dim=1)
        mean_embed_B = torch.mean(token_embed_B, dim=1)
        concat_embeddings = torch.cat(
            [
                mean_embed_A,
                torch.mul(mean_embed_A, mean_embed_B),
                torch.subtract(mean_embed_A, mean_embed_B),
                mean_embed_B,
            ],
            dim=1,
        )
        relation_cls_head = self.relation_cls_head.to(concat_embeddings.device)
        relation_scores = relation_cls_head(concat_embeddings)

        # dists = torch.sqrt(
        #     torch.sum(torch.pow(mean_embed_A - mean_embed_B, 2), dim=1)
        # )

        outputs = {
            "triple_scores": triple_scores,
            "relation_scores": relation_scores,
            # "dists": dists,
        }
        return outputs


class TextScorer(TextEncoder):
    """Score tokenized text with a Transformer LM"""

    def __init__(self, num_entities, num_relations, model_name, pool_strategy):
        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            model_name=model_name,
            pool_strategy=pool_strategy,
        )

        self.hidden_size = self.encoder.config.hidden_size
        self.triple_score_head = torch.nn.Linear(self.hidden_size, 1)
        self.relation_cls_head = torch.nn.Linear(self.hidden_size, self.num_relations)

    def forward(
        self, input_ids, token_type_ids, attention_mask, output_embeddings=False
    ):
        encodings = super().forward(input_ids, token_type_ids, attention_mask)
        triple_scores = self.triple_score_head.to(encodings.device)(encodings)
        relation_scores = self.relation_cls_head.to(encodings.device)(encodings)

        outputs = {"triple_scores": triple_scores, "relation_scores": relation_scores}

        if output_embeddings:
            outputs["embeddings"] = encodings

        return outputs


def _pool_lm(encoder_output, attention_mask, pool_strategy=None):
    """Pool the encoded text from a transformers model"""
    if pool_strategy is None:
        return encoder_output[0]
    if pool_strategy == "mean":
        return _mean_pool_lm(encoder_output, attention_mask)
    elif pool_strategy == "max":
        return _max_pool_lm(encoder_output, attention_mask)
    elif pool_strategy == "cls":
        return _cls_pool_lm(encoder_output)

    raise ValueError(f"pool_strategy={pool_strategy} not supported")


def _mean_pool_lm(encoder_output, attention_mask):
    """Credit: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens"""
    token_embeddings = encoder_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def _max_pool_lm(encoder_output, attention_mask):
    """Credit: https://huggingface.co/sentence-transformers/bert-base-nli-max-tokens"""
    token_embeddings = encoder_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    token_embeddings[
        input_mask_expanded == 0
    ] = -1e9  # Set padding tokens to small negative value
    max_over_time = torch.max(token_embeddings, 1)[0]
    return max_over_time


def _cls_pool_lm(encoder_output):
    """Return the [CLS] hidden states of each input"""
    return encoder_output[0][:, 0]
