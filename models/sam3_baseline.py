from __future__ import annotations

import importlib
from contextlib import nullcontext
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
        self.processor = None
        self.official_backend = False
        self.processor_resolution = 1008
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        if self.allow_mock or self.backend == "mock":
            return MockSAM3()

        errors: List[str] = []
        candidates = []
        if self.backend in {"auto", "sam3"}:
            candidates.extend([
                ("sam3", "build_sam3_image_model"),
                ("sam3.model_builder", "build_sam3_image_model"),
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
                if attr_name == "build_sam3_image_model":
                    self.official_backend = True
                    model = attr(
                        device=self.device_name,
                        checkpoint_path=self.checkpoint,
                        load_from_HF=self.checkpoint is None,
                        eval_mode=True,
                    )
                    processor_mod = importlib.import_module("sam3.model.sam3_image_processor")
                    self.processor = processor_mod.Sam3Processor(
                        model,
                        resolution=self.processor_resolution,
                        device=self.device_name,
                        confidence_threshold=0.0,
                    )
                    return model
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

    def train(self, mode: bool = True) -> "SAM3Adapter":
        super().train(mode)
        if self.official_backend:
            self.model.eval()
        return self

    @property
    def image_encoder(self) -> nn.Module:
        if self.official_backend and hasattr(self.model, "backbone"):
            return self.model.backbone.vision_backbone
        for name in ["image_encoder", "vision_encoder", "encoder"]:
            if hasattr(self.model, name):
                return getattr(self.model, name)
        return self.model

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.official_backend and hasattr(self.model, "backbone"):
            # Official SAM3 uses fused inference kernels in the image trunk that
            # assert autograd is disabled, even when the surrounding FloodRoad
            # training loop is computing gradients for policy/head modules.
            with torch.no_grad():
                image = self._official_transform_tensor(image)
                with self._official_autocast(image.device):
                    out = self.model.backbone.forward_image(image)
            features = out.get("vision_features")
            if features is None and out.get("backbone_fpn"):
                features = out["backbone_fpn"][-1]
            if features is None:
                raise SAM3IntegrationError("Official SAM3 backbone output has no vision_features/backbone_fpn.")
            return features.float()
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
        if self.official_backend and hasattr(self.model, "backbone"):
            texts = [text] if isinstance(text, str) else text
            with torch.no_grad():
                with self._official_autocast(device):
                    out = self.model.backbone.forward_text(texts, device=device)
                embeds = out.get("language_embeds")
                features = out.get("language_features")
                if embeds is not None:
                    pooled = embeds.mean(dim=0)
                elif features is not None:
                    pooled = features.mean(dim=0)
                else:
                    raise SAM3IntegrationError("Official SAM3 text output has no language embeddings/features.")
                return F.normalize(pooled.float(), dim=-1)
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
        if self.official_backend and self.processor is not None:
            return self._official_text_prompt_predict(image, prompt)
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

    def _official_transform_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Transform Bx3xHxW RGB in [0,1] or normalized floats to SAM3 input."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise SAM3IntegrationError("Official SAM3 expects image tensor shaped Bx3xHxW.")
        x = image.float()
        if x.min() < -0.1 or x.max() > 1.5:
            # Inverse ImageNet normalization used by FloodRoadDataset.
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = x * std + mean
        x = x.clamp(0, 1)
        x = F.interpolate(x, size=(self.processor_resolution, self.processor_resolution), mode="bilinear", align_corners=False)
        return (x - 0.5) / 0.5

    def _official_autocast(self, device: torch.device | str):
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return nullcontext()
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        dtype = torch.bfloat16 if bf16_supported else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.inference_mode()
    def _official_text_prompt_predict(self, image: torch.Tensor, prompt: str) -> SAM3Output:
        from PIL import Image
        import numpy as np

        masks, scores = [], []
        for img in image.detach().cpu():
            rgb = img.float()
            if rgb.min() < -0.1 or rgb.max() > 1.5:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                rgb = rgb * std + mean
            rgb = (rgb.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(rgb)
            with self._official_autocast(image.device):
                state = self.processor.set_image(pil)
                out = self.processor.set_text_prompt(prompt=prompt, state=state)
            out_masks = out.get("masks_logits")
            if out_masks is None:
                out_masks = out.get("masks")
            out_scores = out.get("scores")
            if out_masks is None or out_masks.numel() == 0:
                mask = torch.zeros((1, image.shape[-2], image.shape[-1]), device=image.device)
                score = torch.tensor(0.0, device=image.device)
            else:
                if out_scores is not None and out_scores.numel() > 0:
                    idx = int(out_scores.detach().float().argmax().item())
                else:
                    idx = 0
                mask = out_masks[idx].float().to(image.device)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                score = out_scores[idx].float().to(image.device) if out_scores is not None and out_scores.numel() > 0 else torch.tensor(1.0, device=image.device)
            masks.append(mask)
            scores.append(score)
        logits = torch.stack(masks, dim=0)
        if logits.shape[1] != 1:
            logits = logits[:, :1]
        # The processor returns probabilities/logits after sigmoid interpolation; convert probabilities to logits.
        logits = torch.logit(logits.clamp(1e-4, 1 - 1e-4))
        return SAM3Output(logits=logits, masks=(logits.sigmoid() > 0.5), scores=torch.stack(scores))

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
