import torch
import numpy as np

from ogb.linkproppred import Evaluator


def evaluate_ranks(
    dataset,
    scores,
    ks=[1, 3, 10],
    device="cuda",
    filter_splits=["train", "valid", "test"],
    metrics=["mr", "mrr", "hits"],
    return_filtered_scores=False,
    replace_na=False,
):
    scores = scores.clone()

    # All entities considered candidates, so we need to filter
    test_triples = torch.tensor(dataset.triples, dtype=torch.long, device=device)
    all_triples = np.concatenate(
        [dataset.filter_splits[filter_split] for filter_split in filter_splits], axis=0
    )
    all_triples = torch.tensor(all_triples, dtype=torch.long, device=device)

    # Save true subject/object scores
    if dataset.query_mode == "head-batch":
        targets = test_triples[:, 2].long()
    else:
        targets = test_triples[:, 0].long()

    true_scores = scores[torch.arange(len(scores)), targets].clone().view(-1, 1)

    # Filter out scores of all true answers to queries
    scores = filter_false_negatives(
        scores,
        test_triples,
        all_triples,
        query_mode=dataset.query_mode,
        replace_na=replace_na,
    )

    # follow LibKGE protocol: Take mean rank among all entities with same score
    is_close = torch.isclose(scores, true_scores.view(-1, 1), rtol=1e-05, atol=1e-04,)
    is_greater = scores > true_scores.view(-1, 1)
    num_ties = torch.sum(is_close, dim=1, dtype=torch.long)
    rank = torch.sum(is_greater & ~is_close, dim=1, dtype=torch.long)
    ranks = rank + num_ties // 2 + 1  # ranks are one-indexed
    ranks = ranks.double().cpu().numpy()

    outputs = {}

    if "mr" in metrics:
        outputs["mr"] = ranks

    if "mrr" in metrics:
        outputs["mrr"] = 1 / ranks

    if "hits" in metrics:
        for k in ks:
            hits = ranks <= k
            outputs[f"hits@{k}"] = hits

    if return_filtered_scores:
        outputs["filtered_scores"] = scores

    return outputs


def evaluate_ranks_ogb(scores):
    """Get MRR + Hits@{1,3,10} using OGB's evaluation API"""
    evaluator = Evaluator(name="ogbl-biokg")
    y_pred_pos = scores[:, 0]
    y_pred_neg = scores[:, 1:]
    eval_metrics = evaluator.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})

    outputs = {}
    outputs["mrr"] = eval_metrics["mrr_list"].cpu().numpy()
    outputs["mr"] = 1 / outputs["mrr"]  # add raw mean rank
    outputs["hits@1"] = eval_metrics["hits@1_list"].cpu().numpy()
    outputs["hits@3"] = eval_metrics["hits@3_list"].cpu().numpy()
    outputs["hits@10"] = eval_metrics["hits@10_list"].cpu().numpy()
    return outputs


def filter_false_negatives(
    scores, test_triples, all_triples, query_mode="head-batch", replace_na=False
):
    """
    :param scores: scores of test predictions
    :param test_triples: torch.Tensor of test triples
    :param all_triples: torch.Tensor of all triples for filtering
    :return scores: scores for each subject or object in test,
        with false negatives filtered out
    """
    for i, triple in enumerate(test_triples):
        subj, rel, obj = triple

        rel_idx = all_triples[:, 1] == rel
        if query_mode == "head-batch":
            ent_idx = (all_triples[:, 0] == subj) & (all_triples[:, 2] != obj)
            target_idx = 2
        else:
            ent_idx = (all_triples[:, 2] == obj) & (all_triples[:, 0] != subj)
            target_idx = 0

        row = scores[i]
        min_score = torch.min(row[row != float("-Inf")])
        if len(row[row == float("-Inf")]) > 0:
            scores[i][row == float("-Inf")] = min_score - 1.0

        false_neg_idx = all_triples[rel_idx & ent_idx][:, target_idx]

        if replace_na:
            scores[i, false_neg_idx] = min_score - 1.0
            scores[i][row == float("-Inf")] = min_score - 1.0
        else:
            scores[i, false_neg_idx] = float("-Inf")  # ranked last
    return scores
