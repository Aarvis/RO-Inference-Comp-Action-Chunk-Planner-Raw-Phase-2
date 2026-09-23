from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_MODEL_CACHE_DIR = Path(r"E:\Robot-Origami-Challenge\OOI_Zoom_Module\models")
DEFAULT_DINO_REPO_PREFIX = "facebook/"


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(round(dim * mlp_ratio))
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, top_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(top_tokens)
        kv = self.norm_kv(context_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        top_tokens = top_tokens + attn_out
        top_tokens = top_tokens + self.mlp(self.norm_out(top_tokens))
        return top_tokens


class ConvHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OOIZoomModel(nn.Module):
    def __init__(self, config: dict[str, Any], *, load_encoder: bool = True) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = int(config.get("embed_dim", 384))
        self.wrist_tokens_per_view = int(config.get("wrist_tokens_per_view", 64))
        self.use_wrist_inputs = bool(config.get("use_wrist_inputs", False))
        self.freeze_dino = bool(config.get("freeze_dino", True))
        self.encoder: nn.Module | None = None

        if load_encoder:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError("Install transformers to use OOIZoomModel.") from exc

            model_source = resolve_model_source(
                config["dino_model_name"],
                cache_dir=config.get("model_cache_dir"),
                local_files_only=bool(config.get("local_files_only", False)),
            )
            self.encoder = AutoModel.from_pretrained(
                model_source,
                local_files_only=bool(config.get("local_files_only", False)),
                trust_remote_code=bool(config.get("trust_remote_code", False)),
            )
            self.encoder_dim = self._infer_encoder_dim()
        else:
            self.encoder_dim = int(config.get("encoder_dim", self.embed_dim))
        self.input_proj = nn.Linear(self.encoder_dim, self.embed_dim) if self.encoder_dim != self.embed_dim else nn.Identity()
        self.context_proj = nn.Linear(self.embed_dim, self.embed_dim)

        layers = int(config.get("cross_attention_layers", 2))
        heads = int(config.get("attention_heads", 4))
        mlp_ratio = float(config.get("mlp_ratio", 2.0))
        dropout = float(config.get("dropout", 0.1))
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(self.embed_dim, heads, mlp_ratio, dropout) for _ in range(layers)]
        )

        self.heatmap_head = ConvHead(self.embed_dim, 1)
        self.offset_head = ConvHead(self.embed_dim, 2)
        self.size_head = ConvHead(self.embed_dim, 1)
        self.confidence_head = ConvHead(self.embed_dim, 1)

        if self.freeze_dino and self.encoder is not None:
            self.set_dino_trainable(False)
            self.unfreeze_last_blocks(int(config.get("unfreeze_last_n_blocks", 0)))

    def _infer_encoder_dim(self) -> int:
        if self.encoder is None:
            return self.embed_dim
        encoder_config = getattr(self.encoder, "config", None)
        for key in ("hidden_size", "hidden_dim", "embed_dim"):
            value = getattr(encoder_config, key, None)
            if value is not None:
                return int(value)
        return self.embed_dim

    def set_dino_trainable(self, trainable: bool) -> None:
        if self.encoder is None:
            return
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def unfreeze_last_blocks(self, n_blocks: int) -> None:
        if n_blocks <= 0 or self.encoder is None:
            return
        candidates = []
        for attr in ("encoder.layer", "encoder.layers", "layers", "blocks"):
            obj: Any = self.encoder
            ok = True
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok and hasattr(obj, "__len__"):
                candidates = list(obj)
                break
        for block in candidates[-n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("OOIZoomModel.encode() requires an internal encoder; construct with load_encoder=True.")
        outputs = self.encoder(pixel_values=images)
        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            tokens = outputs[0]
        else:
            raise RuntimeError("DINO encoder output does not expose last_hidden_state.")
        tokens = self._patch_tokens(tokens)
        return self.input_proj(tokens)

    @staticmethod
    def _patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
        n_tokens = tokens.shape[1]
        grid = int(math.sqrt(n_tokens))
        if grid * grid == n_tokens:
            return tokens
        grid_without_cls = int(math.sqrt(n_tokens - 1))
        if grid_without_cls * grid_without_cls == n_tokens - 1:
            return tokens[:, 1:, :]
        patch_count = grid * grid
        if patch_count <= 0:
            raise RuntimeError(f"Could not infer patch-token grid from {n_tokens} tokens.")
        return tokens[:, -patch_count:, :]

    def _reduce_wrist_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] <= self.wrist_tokens_per_view:
            return tokens
        tokens_t = tokens.transpose(1, 2)
        reduced = torch.nn.functional.adaptive_avg_pool1d(tokens_t, self.wrist_tokens_per_view)
        return reduced.transpose(1, 2)

    def forward_from_tokens(
        self,
        top_tokens: torch.Tensor,
        left_wrist_tokens: torch.Tensor | None = None,
        right_wrist_tokens: torch.Tensor | None = None,
        *,
        tokens_are_projected: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not tokens_are_projected:
            top_tokens = self.input_proj(top_tokens)
            if left_wrist_tokens is not None:
                left_wrist_tokens = self.input_proj(left_wrist_tokens)
            if right_wrist_tokens is not None:
                right_wrist_tokens = self.input_proj(right_wrist_tokens)
        if self.use_wrist_inputs and left_wrist_tokens is not None and right_wrist_tokens is not None:
            left_tokens = self._reduce_wrist_tokens(left_wrist_tokens)
            right_tokens = self._reduce_wrist_tokens(right_wrist_tokens)
            context = torch.cat([left_tokens, right_tokens], dim=1)
        else:
            context = self._reduce_wrist_tokens(top_tokens)
        context = self.context_proj(context)

        fused = top_tokens
        for block in self.cross_blocks:
            fused = block(fused, context)

        num_tokens = fused.shape[1]
        grid = int(math.sqrt(num_tokens))
        if grid * grid != num_tokens:
            raise RuntimeError(f"Expected square token grid, got {num_tokens} tokens.")
        features = fused.transpose(1, 2).reshape(fused.shape[0], self.embed_dim, grid, grid)
        return {
            "heatmap_logits": self.heatmap_head(features),
            "offset": torch.sigmoid(self.offset_head(features)),
            "size": torch.sigmoid(self.size_head(features)),
            "confidence_logits": self.confidence_head(features),
        }

    def forward(
        self,
        top: torch.Tensor,
        left_wrist: torch.Tensor | None = None,
        right_wrist: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        top_tokens = self.encode(top)
        left_tokens = self.encode(left_wrist) if left_wrist is not None else None
        right_tokens = self.encode(right_wrist) if right_wrist is not None else None
        return self.forward_from_tokens(
            top_tokens,
            left_tokens,
            right_tokens,
            tokens_are_projected=True,
        )


def resolve_model_source(
    model_name_or_path: str,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> str:
    model_name_or_path = normalize_dino_model_name(model_name_or_path)
    path = Path(model_name_or_path)
    if path.exists():
        return str(path)

    cache_root = Path(cache_dir) if cache_dir else DEFAULT_MODEL_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    local_dir = cache_root / model_name_or_path.split("/")[-1]
    if local_dir.exists() and (local_dir / "config.json").exists():
        return str(local_dir)
    if local_files_only:
        raise FileNotFoundError(
            "DINO model directory is required for offline OOI inference but was not found. "
            f"Checked '{path}' and cached snapshot '{local_dir}'."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install huggingface-hub to download DINO model locally.") from exc

    downloaded = snapshot_download(
        repo_id=model_name_or_path,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return downloaded


def normalize_dino_model_name(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    if path.exists() or "/" in model_name_or_path or "\\" in model_name_or_path:
        return model_name_or_path
    if model_name_or_path.startswith("dinov3-"):
        return f"{DEFAULT_DINO_REPO_PREFIX}{model_name_or_path}"
    return model_name_or_path


def _remap_dinov3_encoder_state_keys(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    model_keys = set(model.state_dict().keys())
    remapped: dict[str, torch.Tensor] = {}
    remapped_count = 0
    for key, value in state_dict.items():
        candidates = [key]
        if key.startswith("encoder.model."):
            candidates.append("encoder." + key[len("encoder.model.") :])
        elif key.startswith("encoder."):
            candidates.append("encoder.model." + key[len("encoder.") :])

        new_key = key
        for candidate in candidates:
            if candidate in model_keys:
                new_key = candidate
                break
        if new_key != key:
            remapped_count += 1
        remapped[new_key] = value
    return remapped, remapped_count


def load_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> Any:
    try:
        return model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as original_exc:
        remapped, remapped_count = _remap_dinov3_encoder_state_keys(model, state_dict)
        if remapped_count == 0:
            raise original_exc
        try:
            result = model.load_state_dict(remapped, strict=strict)
        except RuntimeError as remapped_exc:
            raise remapped_exc from original_exc
        print(f"Remapped {remapped_count} DINOv3 encoder checkpoint keys for this Transformers version.")
        return result
