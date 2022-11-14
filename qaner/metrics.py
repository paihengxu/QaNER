from typing import Dict, List

import numpy as np

from qaner.data_utils import Span


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results


# TODO: add metrics over label types
def compute_metrics(
    spans_true_batch: List[Span],
    spans_pred_batch_top_1: List[List[Span]],
    prompt_mapper: Dict[str, str],
    # context_list: List[str],
) -> Dict[str, float]:
    """
    Compute NER metrics.

    Args:
        spans_true_batch (List[Span]): targets.
        spans_pred_batch_top_1 (np.ndarray): predictions.
        prompt_mapper (Dict[str, str]): prompt mapper.
        context_list:
    Returns:
        Dict[str, float]: metrics.
    """

    metrics = {}

    entity_mapper = {"O": 0}
    for entity_tag in prompt_mapper:
        entity_mapper[entity_tag] = len(entity_mapper)

    ner_confusion_matrix = np.zeros((len(entity_mapper), len(entity_mapper)))
    confusion_matrix_true_denominator = np.zeros(len(entity_mapper))
    confusion_matrix_pred_denominator = np.zeros(len(entity_mapper))

    # assert len(context_list) % (len(entity_mapper)-1) == 0, \
    #     f"batch size: {len(context_list)}, # label: {len(entity_mapper)-1}:"

    # batch_label = []
    # batch_pred = []
    for idx, (span_true, span_pred) in enumerate(zip(spans_true_batch, spans_pred_batch_top_1)):
        # if idx % (len(entity_mapper)-1) == 0:
        #     if idx != 0:
        #         batch_label.append(tmp_label)
        #         batch_pred.append(tmp_pred)
        #     else:
        #         context = context_list[idx]
        #         context_len = len(context.split(' '))
        #         tmp_label = ['O'] * context_len
        #         tmp_pred = ['O'] * context_len

        span_pred = span_pred[0]  # type: ignore

        i = entity_mapper[span_true.label]
        j = entity_mapper[span_pred.label]  # type: ignore

        confusion_matrix_true_denominator[i] += 1
        confusion_matrix_pred_denominator[j] += 1

        if span_true == span_pred:
            ner_confusion_matrix[i, j] += 1

    assert (
        confusion_matrix_true_denominator.sum()
        == confusion_matrix_pred_denominator.sum()
    )

    ner_confusion_matrix_diag = np.diag(ner_confusion_matrix)

    # TODO: hide RuntimeWarning
    accuracy = np.nan_to_num(
        ner_confusion_matrix_diag.sum() / confusion_matrix_true_denominator.sum()
    )
    precision_per_entity_type = np.nan_to_num(
        ner_confusion_matrix_diag / confusion_matrix_pred_denominator
    )
    recall_per_entity_type = np.nan_to_num(
        ner_confusion_matrix_diag / confusion_matrix_true_denominator
    )
    f1_per_entity_type = np.nan_to_num(
        2
        * precision_per_entity_type
        * recall_per_entity_type
        / (precision_per_entity_type + recall_per_entity_type)
    )
    metrics["accuracy"] = accuracy

    for label_tag, idx in entity_mapper.items():
        if idx == 0:
            continue
        metrics[f"precision_tag_{label_tag}"] = precision_per_entity_type[idx]
        metrics[f"recall_tag_{label_tag}"] = recall_per_entity_type[idx]
        metrics[f"f1_tag_{label_tag}"] = f1_per_entity_type[idx]

    # TODO: add micro average

    # macro average
    metrics["precision_macro"] = precision_per_entity_type[1:].mean()
    metrics["recall_macro"] = recall_per_entity_type[1:].mean()
    metrics["f1_macro"] = f1_per_entity_type[1:].mean()

    # weighted average
    if np.sum(confusion_matrix_true_denominator[1:]) == 0:
        metrics["precision_weighted"] = 0
        metrics["recall_weighted"] = 0
        metrics["f1_weighted"] = 0
    else:
        metrics["precision_weighted"] = np.average(
            precision_per_entity_type[1:],
            weights=confusion_matrix_true_denominator[1:],
        )
        metrics["recall_weighted"] = np.average(
            recall_per_entity_type[1:],
            weights=confusion_matrix_true_denominator[1:],
        )
        metrics["f1_weighted"] = np.average(
            f1_per_entity_type[1:],
            weights=confusion_matrix_true_denominator[1:],
        )

    return metrics
