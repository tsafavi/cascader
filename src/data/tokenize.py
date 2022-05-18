import torch

from transformers import AutoTokenizer


class TokenizedDataBatch(object):
    """Custom object for batching tokenized text"""

    def __init__(self, input_ids, token_type_ids, attention_mask):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.token_type_ids = self.token_type_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self

    @staticmethod
    def create(data, key=None):
        input_ids = []
        token_type_ids = []
        attention_mask = []

        for example in data:
            if key is not None:
                example = example[key]
            input_ids.append(example["input_ids"])
            token_type_ids.append(example["token_type_ids"])
            attention_mask.append(example["attention_mask"])

        input_ids = torch.vstack(input_ids)
        token_type_ids = torch.vstack(token_type_ids)
        attention_mask = torch.vstack(attention_mask)

        return TokenizedDataBatch(input_ids, token_type_ids, attention_mask)


class Tokenizer(object):
    """Wrapper class for a transformers tokenizer"""

    def __init__(self, model_name, max_length):
        self._base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        self.cls_token = self._base_tokenizer.cls_token
        self.sep_token = self._base_tokenizer.sep_token
        self.cls_token_id = self._base_tokenizer.convert_tokens_to_ids(self.cls_token)
        self.sep_token_id = self._base_tokenizer.convert_tokens_to_ids(self.sep_token)

    def __call__(self, segments, use_relations=False):
        return self._tokenize(segments, use_relations=use_relations)

    def _tokenize(self, segments, use_relations=False):
        num_segments = len(segments)

        # Convert all text segments and special tokens to IDs
        segments_tokenized = [
            self._base_tokenizer.convert_tokens_to_ids(
                self._base_tokenizer.tokenize(
                    segment, max_length=self.max_length, truncation=True
                )
            )
            for segment in segments
        ]

        # Truncate first (and optionally last) text segments
        num_cls = 1
        num_sep = num_segments
        segment_lengths = [len(segment) for segment in segments_tokenized]
        num_rm = max(sum(segment_lengths) + num_cls + num_sep - self.max_length, 0)

        if num_segments > 1:
            pair_ids = segments_tokenized[-1]
        else:
            pair_ids = None

        truncated_first, truncated_second, _ = self._base_tokenizer.truncate_sequences(
            segments_tokenized[0],
            pair_ids=pair_ids,
            num_tokens_to_remove=num_rm,
            truncation_strategy="longest_first",
        )
        segments_tokenized[0] = truncated_first

        if num_segments > 1:
            segments_tokenized[-1] = truncated_second

        # Make sure the truncation worked
        segment_lengths = [len(segment) for segment in segments_tokenized]
        num_pad = self.max_length - num_cls - num_sep - sum(segment_lengths)

        # Construct the input sequence + token type IDs for the first segment
        input_ids = []
        input_ids.append(self.cls_token_id)
        input_ids.extend(segments_tokenized[0])
        input_ids.append(self.sep_token_id)

        if num_segments == 3 and use_relations:  # middle segment = relation
            input_ids.extend(segments_tokenized[1])
            input_ids.append(self.sep_token_id)

        num_segment_a = len(input_ids)
        token_type_ids = [0] * num_segment_a

        # Construct the input sequence + token type IDs for the second segment
        if num_segments == 2 or (num_segments == 3 and use_relations):
            input_ids.extend(segments_tokenized[-1])
            input_ids.append(self.sep_token_id)

            num_segment_b = len(input_ids) - num_segment_a
            token_type_ids += [1] * num_segment_b

        num_true_inputs = len(input_ids)

        # Pad the input sequence + token type IDs
        input_ids += [0] * num_pad
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long)
        token_type_ids += [0] * num_pad
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

        # Construct the attention mask
        attention_mask = [1] * num_true_inputs + [0] * num_pad
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        outputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

        return outputs
