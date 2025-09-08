from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.trainer import EvalPrediction


def compute_ppl(
    eval_prediction: EvalPrediction, _: Optional[PreTrainedTokenizer] = None
) -> torch.Tensor:
    logits = eval_prediction.predictions
    labels_ids = eval_prediction.label_ids

    return torch.exp(
        F.cross_entropy(logits, labels_ids, reduction="mean", ignore_index=-100)
    )


def compute_ttr(
    eval_prediction: EvalPrediction, tokenizer: PreTrainedTokenizer
) -> torch.Tensor:
    logits = eval_prediction.predictions

    prediction = logits.argmax(dim=-1)
    prediction_text = tokenizer.decode(prediction)

    prediction_words = prediction_text.split()
    prediction_unique_words = set(prediction_words)
    return (
        len(prediction_unique_words) / len(prediction_words)
        if len(prediction_words) > 0
        else 0.0
    )


def compute_entropy(
    eval_prediction: EvalPrediction, _: Optional[PreTrainedTokenizer] = None
) -> torch.Tensor:
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids

    mask = labels != -100

    token_entropy = -torch.sum(
        torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1), dim=-1
    )

    return token_entropy.sum() / mask.sum()

SFT_EVAL_METRICS_TO_FN = {
    "ppl": compute_ppl,
    "ttr": compute_ttr,
    "entropy": compute_entropy,
}


def get_compute_metrics_fn(
    eval_metrics: Optional[str] = None, tokenizer: Optional[PreTrainedTokenizer] = None
) -> Optional[Callable[[EvalPrediction], Dict[str, torch.Tensor]]]:
    if eval_metrics is None:
        return None

    metrics = eval_metrics.split(",")

    if "ttr" in metrics and tokenizer is None:
        raise ValueError("Tokenizer is required for ttr metric")

    def _compute_metrics(eval_prediction: EvalPrediction) -> Dict[str, torch.Tensor]:
        output = {}
        for metric in metrics:
            if metric not in SFT_EVAL_METRICS_TO_FN:
                raise ValueError(f"Invalid metric: {metric}")

            output[metric] = SFT_EVAL_METRICS_TO_FN[metric](eval_prediction, tokenizer)

        return output

    return _compute_metrics
