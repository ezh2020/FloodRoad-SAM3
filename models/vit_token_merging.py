from __future__ import annotations

from types import MethodType
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


def apply_vit_token_merging(
    image_encoder: nn.Module,
    merge_ratio: float = 0.2,
    layers: Optional[Iterable[int]] = None,
    metric_dim: int = 64,
    merge_attention: bool = True,
    merge_mlp: bool = True,
) -> List[str]:
    """Patch SAM3 ViT blocks with ToMe-style token merging inside each block.

    Official SAM3's ViTDet attention uses RoPE buffers tied to the native spatial
    grid. Reducing the image resolution or leaving a block with fewer tokens can
    break those shape assumptions. This patch keeps attention and all external
    block inputs/outputs at the original HxW shape, but runs the block MLP on a
    merged token set and unmerges the MLP output before the residual add.
    """

    trunk = getattr(image_encoder, "trunk", image_encoder)
    blocks = getattr(trunk, "blocks", None)
    if blocks is None:
        return []

    layer_set = {int(v) for v in (layers or [])}
    if not layer_set:
        layer_set = set(range(1, len(blocks) + 1))

    patched: List[str] = []
    for idx, block in enumerate(blocks):
        layer_id = idx + 1
        if idx not in layer_set:
            continue
        if not all(hasattr(block, name) for name in ["norm1", "attn", "norm2", "mlp", "ls1", "ls2"]):
            continue
        original_forward = getattr(block, "forward")
        globals_ = getattr(getattr(original_forward, "__func__", original_forward), "__globals__", {})
        window_partition = globals_.get("window_partition")
        window_unpartition = globals_.get("window_unpartition")
        block.forward = MethodType(
            _make_merged_block_forward(
                merge_ratio=float(merge_ratio),
                metric_dim=int(metric_dim),
                merge_attention=bool(merge_attention),
                merge_mlp=bool(merge_mlp),
                window_partition=window_partition,
                window_unpartition=window_unpartition,
            ),
            block,
        )
        block._floodroad_token_merge = {
            "merge_ratio": float(merge_ratio),
            "metric_dim": int(metric_dim),
            "merge_attention": bool(merge_attention),
            "merge_mlp": bool(merge_mlp),
        }
        patched.append(f"blocks.{idx}")
    return patched


def _make_merged_block_forward(
    merge_ratio: float,
    metric_dim: int,
    merge_attention: bool,
    merge_mlp: bool,
    window_partition,
    window_unpartition,
):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_norm = self.norm1(x)
        if getattr(self, "window_size", 0) > 0:
            if window_partition is None or window_unpartition is None:
                raise RuntimeError("SAM3 window attention helpers are unavailable for token merging.")
            h, w = x_norm.shape[1], x_norm.shape[2]
            x_attn, pad_hw = window_partition(x_norm, self.window_size)
        else:
            h = w = None
            x_attn = x_norm
            pad_hw = None

        if merge_attention:
            x_attn = _merged_attention(self.attn, x_attn, merge_ratio, metric_dim)
        else:
            x_attn = self.attn(x_attn)
        x_attn = self.ls1(x_attn)
        if getattr(self, "window_size", 0) > 0:
            x_attn = window_unpartition(x_attn, self.window_size, pad_hw, (h, w))

        x = shortcut + self.dropout(self.drop_path(x_attn))
        mlp_out = _merged_mlp(self, x, merge_ratio, metric_dim) if merge_mlp else self.mlp(self.norm2(x))
        x = x + self.dropout(self.drop_path(self.ls2(mlp_out)))
        return x

    return forward


def _merged_attention(attn: nn.Module, x: torch.Tensor, merge_ratio: float, metric_dim: int) -> torch.Tensor:
    if merge_ratio <= 0 or x.ndim != 4 or getattr(attn, "use_rel_pos", False):
        return attn(x)
    b, h, w, c = x.shape
    n = h * w
    r = int(round(n * merge_ratio))
    if r <= 0:
        return attn(x)

    qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, -1)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn._apply_rope(q, k)

    metric = k.mean(dim=1)[..., : max(1, min(metric_dim, k.shape[-1]))]
    merge, unmerge = bipartite_soft_matching(metric, r)
    q = _merge_heads(merge, q)
    k = _merge_heads(merge, k)
    v = _merge_heads(merge, v)

    x_attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x_attn = x_attn.transpose(1, 2).reshape(b, -1, c)
    x_attn = attn.proj(x_attn)
    return unmerge(x_attn).reshape(b, h, w, c)


def _merge_heads(merge, x: torch.Tensor) -> torch.Tensor:
    b, heads, tokens, channels = x.shape
    flat = x.transpose(1, 2).reshape(b, tokens, heads * channels)
    flat = merge(flat, mode="mean")
    return flat.reshape(b, -1, heads, channels).transpose(1, 2)


def _merged_mlp(block: nn.Module, x: torch.Tensor, merge_ratio: float, metric_dim: int) -> torch.Tensor:
    if merge_ratio <= 0 or x.ndim != 4:
        return block.mlp(block.norm2(x))
    b, h, w, c = x.shape
    flat = x.reshape(b, h * w, c)
    r = int(round(flat.shape[1] * merge_ratio))
    merge, unmerge = bipartite_soft_matching(flat[..., : max(1, min(metric_dim, c))], r)
    merged = merge(flat, mode="mean")
    mlp_out = block.mlp(block.norm2(merged))
    return unmerge(mlp_out).reshape(b, h, w, c)


def bipartite_soft_matching(metric: torch.Tensor, r: int) -> Tuple:
    """Small self-contained subset of facebookresearch/ToMe matching."""
    n_tokens = metric.shape[1]
    r = min(max(int(r), 0), n_tokens // 2)
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        src_metric = metric[..., ::2, :]
        dst_metric = metric[..., 1::2, :]
        scores = src_metric @ dst_metric.transpose(-1, -2)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src = x[..., ::2, :]
        dst = x[..., 1::2, :]
        bsz, src_tokens, channels = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(bsz, src_tokens - r, channels))
        src = src.gather(dim=-2, index=src_idx.expand(bsz, r, channels))
        dst = dst.scatter_reduce(-2, dst_idx.expand(bsz, r, channels), src, reduce=mode)
        if n_tokens % 2 == 1:
            dst = torch.cat([dst, x[..., -1:, :]], dim=-2)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = n_tokens // 2
        if n_tokens % 2 == 1:
            dst_len += 1
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]
        tail = None
        if n_tokens % 2 == 1:
            tail = dst[..., -1:, :]
            dst = dst[..., :-1, :]
        bsz, _, channels = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(bsz, r, channels))
        out = torch.zeros(bsz, n_tokens, channels, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(bsz, unm_len, channels), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(bsz, r, channels), src=src)
        if tail is not None:
            out[..., -1:, :] = tail
        return out

    return merge, unmerge


def do_nothing(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    return x
