from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import torch


def binary_confusion(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.bool()
    target = target.bool()
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def f1_iou_precision_recall(conf: Dict[str, float]) -> Dict[str, float]:
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1.0)
    return {"Pixel F1": f1, "Pixel IoU": iou, "Precision": precision, "Recall": recall}


def accumulate_confusion(total: Dict[str, float], conf: Dict[str, float]) -> Dict[str, float]:
    for key in ["tp", "fp", "fn", "tn"]:
        total[key] = total.get(key, 0.0) + conf.get(key, 0.0)
    return total


def segment_confusion_from_mask(pred_mask: torch.Tensor, graph: nx.Graph, segment_map: torch.Tensor) -> Dict[str, float]:
    pred = pred_mask.detach().cpu().float()
    seg = segment_map.detach().cpu().long()
    tp = fp = fn = tn = 0
    for node, data in graph.nodes(data=True):
        pix = seg == int(node)
        if pix.sum().item() == 0:
            continue
        pred_label = int(pred[pix].mean().item() >= 0.5)
        gt_label = int(data.get("label", 0))
        if pred_label == 1 and gt_label == 1:
            tp += 1
        elif pred_label == 1 and gt_label == 0:
            fp += 1
        elif pred_label == 0 and gt_label == 1:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body)

