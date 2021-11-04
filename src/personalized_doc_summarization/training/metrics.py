import sklearn.metrics
import numpy as np
from rouge_score import rouge_scorer


def calculate_optimal_F1(targets, scores) -> float:
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        targets, scores, drop_intermediate=False
    )
    auc = sklearn.metrics.auc(fpr, tpr)
    num_pos = targets.sum()
    num_neg = (1 - targets).sum()
    if num_pos ==0:
        return {}
    f1 = 2 * tpr / (1 + tpr + fpr * num_neg / num_pos)
    precision = tpr/(tpr + fpr * num_neg / num_pos)
    recall = list(map(float, list(tpr)))
    precision = list(map(float, list(precision)))
    threshold = thresholds[np.argmax(f1)]
    return {"F1": float(np.max(f1)), "threshold": float(threshold), "AUC": float(auc),
            "Recall": float(recall[np.argmax(f1)]), "Precision": float(precision[np.argmax(f1)])}


def calculate_rouge_scores(target_summary, predicted_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    scores = scorer.score(target_summary, predicted_summary)
    # TODO: determine which scores to pass back
    return scores
