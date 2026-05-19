from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PASSABLE = 0
FLOODED = 1
UNCERTAIN = 2


class SegmentPolicyNet(nn.Module):
    def __init__(self, state_dim: int = 12, hidden_dim: int = 256, num_actions: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


@dataclass
class PolicyRollout:
    actions: torch.Tensor
    logits: torch.Tensor
    log_probs: torch.Tensor
    states: torch.Tensor
    node_ids: List[int]


def build_segment_states(
    score_map: torch.Tensor,
    segment_map: torch.Tensor,
    graph: nx.Graph,
    post_raw: torch.Tensor,
    pre_raw: torch.Tensor,
    decided: Optional[Dict[int, int]] = None,
) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
    decided = decided or {}
    device = score_map.device
    if score_map.ndim == 3:
        score_map = score_map[0]
    node_ids = sorted(int(n) for n in graph.nodes())
    states = []
    evidence = []
    for node in node_ids:
        pix = segment_map == node
        if pix.sum() == 0:
            e = torch.tensor(0.0, device=device)
            mean_rgb = torch.zeros(3, device=device)
            diff_rgb = torch.zeros(3, device=device)
        else:
            e = score_map[pix].mean()
            mean_rgb = post_raw[:, pix].mean(dim=1)
            diff_rgb = (post_raw[:, pix] - pre_raw[:, pix]).abs().mean(dim=1)
        data = graph.nodes[node]
        length = torch.tensor(float(data.get("length", pix.sum().item())), device=device)
        width = torch.tensor(float(data.get("width", 8.0)), device=device)
        length = torch.log1p(length) / 10.0
        width = width / 32.0
        counts = torch.zeros(3, device=device)
        for nb in graph.neighbors(node):
            label = decided.get(int(nb))
            if label is None:
                counts[2] += 1
            elif label == FLOODED:
                counts[1] += 1
            else:
                counts[0] += 1
        denom = counts.sum().clamp_min(1.0)
        counts = counts / denom
        state = torch.cat([e.view(1), length.view(1), width.view(1), mean_rgb, diff_rgb, counts], dim=0)
        states.append(state)
        evidence.append(e)
    if not states:
        return torch.empty(0, 12, device=device), [], torch.empty(0, device=device)
    return torch.stack(states, dim=0), node_ids, torch.stack(evidence, dim=0)


def supervised_segment_targets(graph: nx.Graph, node_ids: List[int], device: torch.device) -> torch.Tensor:
    labels = [int(graph.nodes[n].get("label", 0)) for n in node_ids]
    return torch.tensor(labels, dtype=torch.long, device=device)


def rollout_policy(
    policy: SegmentPolicyNet,
    score_map: torch.Tensor,
    segment_map: torch.Tensor,
    graph: nx.Graph,
    post_raw: torch.Tensor,
    pre_raw: torch.Tensor,
    sample: bool = True,
) -> PolicyRollout:
    with torch.no_grad():
        init_states, init_nodes, evidence = build_segment_states(score_map, segment_map, graph, post_raw, pre_raw)
    order = torch.argsort((evidence - 0.5).abs(), descending=True).tolist()
    decided: Dict[int, int] = {}
    all_states, all_logits, all_actions, all_log_probs, ordered_nodes = [], [], [], [], []
    for idx in order:
        node = init_nodes[idx]
        states, node_ids, _ = build_segment_states(score_map, segment_map, graph, post_raw, pre_raw, decided)
        local_idx = node_ids.index(node)
        state = states[local_idx].detach()
        logits = policy(state.unsqueeze(0)).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if sample else torch.argmax(logits)
        decided[node] = int(action.item())
        all_states.append(state)
        all_logits.append(logits)
        all_actions.append(action)
        all_log_probs.append(dist.log_prob(action))
        ordered_nodes.append(node)

    if not all_states:
        device = score_map.device
        return PolicyRollout(
            actions=torch.empty(0, dtype=torch.long, device=device),
            logits=torch.empty(0, 3, device=device),
            log_probs=torch.empty(0, device=device),
            states=torch.empty(0, 12, device=device),
            node_ids=[],
        )
    return PolicyRollout(
        actions=torch.stack(all_actions),
        logits=torch.stack(all_logits),
        log_probs=torch.stack(all_log_probs),
        states=torch.stack(all_states),
        node_ids=ordered_nodes,
    )


def compute_reward(
    graph: nx.Graph,
    node_ids: List[int],
    actions: torch.Tensor,
    reachability_pairs: int = 50,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weights = weights or {"local": 1.0, "reach": 0.6, "inconsistency": 0.3, "uncertain": 0.1}
    device = actions.device
    pred = {int(n): int(a) for n, a in zip(node_ids, actions.detach().cpu().tolist())}
    gt = {int(n): int(graph.nodes[n].get("label", 0)) for n in graph.nodes()}
    if not pred:
        return torch.tensor(0.0, device=device), {"local": 0.0, "reach": 0.0, "inconsistency": 0.0, "uncertain": 0.0}

    correct = 0
    uncertain = 0
    for n, action in pred.items():
        if action == UNCERTAIN:
            uncertain += 1
        final = FLOODED if action == FLOODED else PASSABLE
        correct += int(final == gt.get(n, PASSABLE))
    r_loc = correct / max(len(pred), 1)
    r_unc = uncertain / max(len(pred), 1)

    edge_count = max(graph.number_of_edges(), 1)
    incon = 0
    for a, b in graph.edges():
        pa = FLOODED if pred.get(int(a), PASSABLE) == FLOODED else PASSABLE
        pb = FLOODED if pred.get(int(b), PASSABLE) == FLOODED else PASSABLE
        incon += int(pa != pb)
    r_incon = incon / edge_count

    r_reach = reachability_agreement(graph, pred, gt, reachability_pairs)
    total = (
        weights.get("local", 1.0) * r_loc
        + weights.get("reach", 0.6) * r_reach
        - weights.get("inconsistency", 0.3) * r_incon
        - weights.get("uncertain", 0.1) * r_unc
    )
    details = {"local": r_loc, "reach": r_reach, "inconsistency": r_incon, "uncertain": r_unc, "total": total}
    return torch.tensor(total, dtype=torch.float32, device=device), details


def reachability_agreement(graph: nx.Graph, pred: Dict[int, int], gt: Dict[int, int], pairs: int) -> float:
    nodes = [int(n) for n in graph.nodes()]
    if len(nodes) < 2:
        return 1.0
    rng = random.Random(0)
    sampled = [(rng.choice(nodes), rng.choice(nodes)) for _ in range(pairs)]
    gt_graph = graph.subgraph([n for n in nodes if gt.get(n, PASSABLE) == PASSABLE]).copy()
    pred_graph = graph.subgraph([n for n in nodes if pred.get(n, PASSABLE) != FLOODED]).copy()
    agree = 0
    for src, dst in sampled:
        gt_ok = src in gt_graph and dst in gt_graph and nx.has_path(gt_graph, src, dst)
        pred_ok = src in pred_graph and dst in pred_graph and nx.has_path(pred_graph, src, dst)
        agree += int(gt_ok == pred_ok)
    return agree / max(len(sampled), 1)


def mask_from_segment_actions(segment_map: torch.Tensor, node_ids: List[int], actions: torch.Tensor, score_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    mask = torch.zeros_like(segment_map, dtype=torch.float32)
    for node, action in zip(node_ids, actions):
        act = int(action.item())
        if act == FLOODED:
            mask[segment_map == int(node)] = 1.0
        elif act == UNCERTAIN and score_map is not None:
            pix = segment_map == int(node)
            if pix.any() and score_map[pix].mean() > 0.5:
                mask[pix] = 1.0
    return mask
