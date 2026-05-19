from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM3IntegrationError(RuntimeError):
    pass


@dataclass
class SAM3Output:
    logits: torch.Tensor
    masks: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None


class MockSAM3(nn.Module):
    """Small deterministic fallback for smoke tests only.

    This is not a scientific baseline. It exists so the rest of the training and
    evaluation pipeline can be syntax-checked before the official SAM3 package is
    installed in Colab.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def decode_mask(self, features: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        logits = self.decoder(features)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor, text: str = "flooded road") -> SAM3Output:
        features = self.encode_image(image)
        logits = self.decode_mask(features, image.shape[-2:])
        return SAM3Output(logits=logits, features=features)


class SAM3Adapter(nn.Module):
    """Thin adapter around official SAM3 or a smoke-test mock backend.

    The adapter exposes image encoding, text/image embedding hooks, and text-only
    mask prediction. When an installed SAM3 package does not match one of the
    known layouts, the raised error explains where to add the project-specific
    glue without touching the rest of the codebase.
    """

    def __init__(
        self,
        backend: str = "auto",
        checkpoint: Optional[str] = None,
        model_type: Optional[str] = None,
        device: str | torch.device = "cuda",
        allow_mock: bool = False,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device_name = str(device)
        self.allow_mock = allow_mock
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        if self.allow_mock or self.backend == "mock":
            return MockSAM3()

        errors: List[str] = []
        candidates = []
        if self.backend in {"auto", "sam3"}:
            candidates.extend([
                ("sam3", "build_sam3"),
                ("segment_anything_3", "build_sam3"),
                ("segment_anything_3", "sam_model_registry"),
            ])
        if self.backend not in {"auto", "sam3"}:
            candidates.append((self.backend, "build_sam3"))

        for module_name, attr_name in candidates:
            try:
                mod = importlib.import_module(module_name)
                attr = getattr(mod, attr_name)
                if attr_name == "sam_model_registry":
                    if self.model_type is None:
                        raise SAM3IntegrationError("sam_model_registry requires sam3.model_type in config")
                    return attr[self.model_type](checkpoint=self.checkpoint)
                try:
                    return attr(checkpoint=self.checkpoint, model_type=self.model_type)
                except TypeError:
                    try:
                        return attr(self.checkpoint)
                    except TypeError:
                        return attr()
            except Exception as exc:
                errors.append(f"{module_name}.{attr_name}: {exc}")

        joined = "\n".join(errors)
        raise SAM3IntegrationError(
            "Could not construct official SAM3 backend. Install Meta's SAM3 package in Colab "
            "and, if its API differs, add the entry point in models/sam3_baseline.py. "
            "For pipeline smoke tests only set sam3.allow_mock=true. Tried:\n" + joined
        )

    @property
    def image_encoder(self) -> nn.Module:
        for name in ["image_encoder", "vision_encoder", "encoder"]:
            if hasattr(self.model, name):
                return getattr(self.model, name)
        return self.model

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "encode_image"):
            return self.model.encode_image(image)
        if hasattr(self.model, "image_encoder"):
            return self.model.image_encoder(image)
        if hasattr(self.model, "vision_encoder"):
            return self.model.vision_encoder(image)
        if hasattr(self.model, "encoder"):
            return self.model.encoder(image)
        raise SAM3IntegrationError("SAM3 backend does not expose an image encoder.")

    def encode_text(self, text: str | List[str], device: torch.device) -> torch.Tensor:
        if hasattr(self.model, "encode_text"):
            emb = self.model.encode_text(text)
            return emb if torch.is_tensor(emb) else torch.as_tensor(emb, device=device)
        texts = [text] if isinstance(text, str) else text
        # Deterministic fallback embedding from text bytes. This is only used when
        # the backend lacks a text encoder; it lets DCA fusion code keep running.
        vals = []
        for item in texts:
            encoded = torch.tensor(list(item.encode("utf-8"))[:64], dtype=torch.float32, device=device)
            if encoded.numel() < 64:
                encoded = F.pad(encoded, (0, 64 - encoded.numel()))
            vals.append(F.normalize(encoded, dim=0))
        return torch.stack(vals, dim=0)

    def encode_patch_embedding(self, patches: torch.Tensor) -> torch.Tensor:
        features = self.encode_image(patches)
        if features.ndim == 4:
            return F.normalize(features.mean(dim=(-2, -1)), dim=-1)
        if features.ndim == 3:
            return F.normalize(features.mean(dim=1), dim=-1)
        return F.normalize(features, dim=-1)

    def decode_mask(self, features: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        if hasattr(self.model, "decode_mask"):
            return self.model.decode_mask(features, output_size)
        for name in ["mask_decoder", "decoder"]:
            if hasattr(self.model, name):
                decoder = getattr(self.model, name)
                try:
                    logits = decoder(features)
                except TypeError:
                    logits = decoder(image_embeddings=features)
                if isinstance(logits, dict):
                    logits = logits.get("logits") or logits.get("masks")
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if logits.ndim == 3:
                    logits = logits.unsqueeze(1)
                return F.interpolate(logits.float(), size=output_size, mode="bilinear", align_corners=False)
        if features.ndim == 4:
            logits = features.mean(dim=1, keepdim=True)
            return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        raise SAM3IntegrationError("SAM3 backend does not expose a compatible mask decoder.")

    def text_prompt_predict(self, image: torch.Tensor, prompt: str) -> SAM3Output:
        if hasattr(self.model, "predict"):
            out = self.model.predict(image=image, text=prompt)
            return normalize_sam_output(out, image.shape[-2:])
        if hasattr(self.model, "forward") and not isinstance(self.model, MockSAM3):
            try:
                out = self.model(image, text=prompt)
                return normalize_sam_output(out, image.shape[-2:])
            except TypeError:
                pass
        features = self.encode_image(image)
        logits = self.decode_mask(features, image.shape[-2:])
        return SAM3Output(logits=logits, features=features)

    def forward(self, image: torch.Tensor, prompt: str = "flooded road") -> SAM3Output:
        return self.text_prompt_predict(image, prompt)


def normalize_sam_output(out: Any, output_size: Tuple[int, int]) -> SAM3Output:
    if isinstance(out, SAM3Output):
        return out
    if torch.is_tensor(out):
        logits = out
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        logits = F.interpolate(logits.float(), size=output_size, mode="bilinear", align_corners=False)
        return SAM3Output(logits=logits)
    if isinstance(out, dict):
        masks = out.get("masks")
        if masks is None:
            masks = out.get("mask")
        if masks is None:
            masks = out.get("logits")
        scores = out.get("scores")
        if scores is None:
            scores = out.get("iou_predictions")
        features = out.get("features")
        if features is None:
            features = out.get("image_embeddings")
        if masks is None:
            raise SAM3IntegrationError("SAM3 output dict has no masks/logits field.")
        logits = masks if torch.is_tensor(masks) else torch.as_tensor(masks)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if logits.ndim == 5:
            # B, K, 1, H, W: pick highest score when available.
            if scores is not None:
                score_t = scores if torch.is_tensor(scores) else torch.as_tensor(scores, device=logits.device)
                idx = score_t.argmax(dim=1)
                logits = logits[torch.arange(logits.shape[0], device=logits.device), idx]
            else:
                logits = logits[:, 0]
        elif logits.ndim == 4 and logits.shape[1] > 1:
            if scores is not None:
                score_t = scores if torch.is_tensor(scores) else torch.as_tensor(scores, device=logits.device)
                idx = score_t.argmax(dim=1)
                logits = logits[torch.arange(logits.shape[0], device=logits.device), idx].unsqueeze(1)
            else:
                logits = logits[:, :1]
        logits = F.interpolate(logits.float(), size=output_size, mode="bilinear", align_corners=False)
        return SAM3Output(logits=logits, masks=logits, scores=scores, features=features)
    if isinstance(out, (tuple, list)):
        return normalize_sam_output(out[0], output_size)
    raise SAM3IntegrationError(f"Unsupported SAM3 output type: {type(out)!r}")


def build_sam3_adapter(cfg: Dict[str, Any]) -> SAM3Adapter:
    return SAM3Adapter(
        backend=cfg.get("backend", "auto"),
        checkpoint=cfg.get("checkpoint"),
        model_type=cfg.get("model_type"),
        device=cfg.get("device", "cuda"),
        allow_mock=bool(cfg.get("allow_mock", False)),
    )
