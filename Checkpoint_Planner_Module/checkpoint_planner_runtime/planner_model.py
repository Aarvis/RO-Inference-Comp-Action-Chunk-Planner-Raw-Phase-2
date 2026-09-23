from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    device = q.device
    dtype = q.dtype
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(dtype=dtype)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(dtype=dtype)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


def build_state_belief(checkpoint_probs: torch.Tensor, phase_logits: torch.Tensor) -> torch.Tensor:
    batch_size, checkpoint_classes = checkpoint_probs.shape
    num_checkpoints = checkpoint_classes - 1
    phase_probs = torch.softmax(phase_logits, dim=-1)
    state = checkpoint_probs.new_zeros((batch_size, (2 * num_checkpoints) + 1))
    state[:, : 2 * num_checkpoints] = (checkpoint_probs[:, :num_checkpoints].unsqueeze(-1) * phase_probs).reshape(
        batch_size, 2 * num_checkpoints
    )
    state[:, 2 * num_checkpoints] = checkpoint_probs[:, num_checkpoints]
    return state


def apply_continuity_prior(
    checkpoint_logits: torch.Tensor,
    last_completed_checkpoint: torch.Tensor,
    gamma: float,
    w_min: float,
    w_future: float,
    lambda_prior: float,
    eps: float = 1e-6,
    gamma_future: float | None = None,
    current_ref_weight: float = 0.3,
    near_future_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, checkpoint_classes = checkpoint_logits.shape
    num_checkpoints = checkpoint_classes - 1
    device = checkpoint_logits.device
    checkpoint_ids = torch.arange(1, num_checkpoints + 1, device=device, dtype=torch.float32).view(1, -1)
    c_ref = last_completed_checkpoint.to(torch.float32).view(-1, 1).clamp(min=0.0, max=float(num_checkpoints))
    weights = torch.full((batch_size, checkpoint_classes), float(w_future), device=device, dtype=checkpoint_logits.dtype)
    checkpoint_weights = weights[:, :num_checkpoints]

    done_mask = c_ref.squeeze(1) >= num_checkpoints
    not_done_mask = ~done_mask
    use_future_decay = gamma_future is not None

    past_ratio_denominator = c_ref.clamp_min(1.0)
    past_ratios = checkpoint_ids / past_ratio_denominator
    past_weights = float(w_min) + (float(current_ref_weight) - float(w_min)) * torch.pow(
        past_ratios.clamp(max=1.0),
        float(gamma),
    )
    past_mask = (c_ref > 0) & (checkpoint_ids < c_ref)
    checkpoint_weights = torch.where(past_mask, past_weights.to(dtype=checkpoint_logits.dtype), checkpoint_weights)

    current_ref_mask = (c_ref > 0) & (checkpoint_ids == c_ref)
    checkpoint_weights = torch.where(
        current_ref_mask,
        torch.full_like(checkpoint_weights, float(current_ref_weight)),
        checkpoint_weights,
    )

    active_ids = c_ref + 1.0
    active_mask = not_done_mask.view(-1, 1) & (checkpoint_ids == active_ids)
    checkpoint_weights = torch.where(active_mask, torch.ones_like(checkpoint_weights), checkpoint_weights)

    near_future_ids = c_ref + 2.0
    near_future_mask = not_done_mask.view(-1, 1) & (checkpoint_ids == near_future_ids)
    checkpoint_weights = torch.where(
        near_future_mask,
        torch.full_like(checkpoint_weights, float(near_future_weight)),
        checkpoint_weights,
    )

    if use_future_decay:
        future_mask = not_done_mask.view(-1, 1) & (checkpoint_ids > near_future_ids)
        future_ratios = near_future_ids / checkpoint_ids
        future_weights = float(w_future) + (float(near_future_weight) - float(w_future)) * torch.pow(
            future_ratios.clamp(max=1.0),
            float(gamma_future),
        )
        checkpoint_weights = torch.where(future_mask, future_weights.to(dtype=checkpoint_logits.dtype), checkpoint_weights)

    weights[:, :num_checkpoints] = checkpoint_weights
    weights[done_mask, num_checkpoints] = 1.0

    final_logits = checkpoint_logits + float(lambda_prior) * torch.log(weights.clamp_min(eps))
    final_probs = torch.softmax(final_logits, dim=-1)
    return final_logits, final_probs


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ff = FeedForward(dim=dim, hidden_dim=ff_hidden_dim, dropout=dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.q_norm(query), self.kv_norm(context), self.kv_norm(context), need_weights=False)
        query = query + self.dropout(attn_out)
        return self.ff(query)


class WristResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        num_queries: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_blocks: int,
        dropout: float,
        num_cameras: int = 2,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.camera_embedding = nn.Embedding(num_cameras, dim)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(dim=dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

    def forward(self, wrist_tokens: torch.Tensor, camera_index: int) -> torch.Tensor:
        batch_size = wrist_tokens.shape[0]
        camera_embed = self.camera_embedding.weight[camera_index].view(1, 1, -1)
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1) + camera_embed
        context = wrist_tokens + camera_embed
        for block in self.blocks:
            query = block(query, context)
        return query


class Pooler(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(self.q_norm(query), self.kv_norm(tokens), self.kv_norm(tokens), need_weights=False)
        return pooled[:, 0, :]


class RotarySelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, rope_base: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary(q, k, positions=positions, base=self.rope_base)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
        key_mask = valid_mask.view(batch_size, 1, 1, seq_len)
        query_mask = valid_mask.view(batch_size, 1, seq_len, 1)
        keep_mask = causal & key_mask & query_mask
        attn_scores = attn_scores.masked_fill(~keep_mask, torch.finfo(attn_scores.dtype).min)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.where(torch.isnan(attn_probs), torch.zeros_like(attn_probs), attn_probs)
        attn_probs = self.attn_dropout(attn_probs)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        attn_out = self.proj_dropout(self.out(attn_out))
        attn_out = attn_out * valid_mask.unsqueeze(-1).to(dtype=attn_out.dtype)
        return residual + attn_out


class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int, dropout: float, rope_base: float) -> None:
        super().__init__()
        self.attn = RotarySelfAttention(dim=dim, num_heads=num_heads, dropout=dropout, rope_base=rope_base)
        self.ff = FeedForward(dim=dim, hidden_dim=ff_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, positions=positions, valid_mask=valid_mask)
        x = self.ff(x)
        return x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)


class TemporalTransformer(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, ff_hidden_dim: int, dropout: float, rope_base: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, positions=positions, valid_mask=valid_mask)
        return self.final_norm(x)


class FrameEncoder(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        state_mean: list[float],
        state_std: list[float],
        tactile_mean: list[float] | None = None,
        tactile_std: list[float] | None = None,
    ) -> None:
        super().__init__()
        model_cfg = config["model"]
        dropout = float(model_cfg.get("dropout", 0.1))
        patch_dim = int(model_cfg.get("patch_dim", 384))
        resampler_queries = int(model_cfg.get("wrist_resampler_queries", 128))
        resampler_blocks = int(model_cfg.get("wrist_resampler_blocks", 2))
        resampler_heads = int(model_cfg.get("attention_heads", 6))
        resampler_ff = int(model_cfg.get("resampler_ffn_dim", 1536))
        state_dim = int(model_cfg.get("state_dim", len(state_mean)))
        state_hidden = int(model_cfg.get("state_hidden_dim", 256))
        state_out = int(model_cfg.get("state_out_dim", 128))
        tactile_dim = int(model_cfg.get("tactile_dim", len(tactile_mean or [])))
        tactile_hidden = int(model_cfg.get("tactile_hidden_dim", 128))
        tactile_out = int(model_cfg.get("tactile_out_dim", 64)) if tactile_dim > 0 else 0
        frame_dim = int(model_cfg.get("frame_dim", 512))
        fusion_hidden = int(model_cfg.get("fusion_hidden_dim", 1024))
        fusion_blocks = int(model_cfg.get("ooi_wrist_fusion_blocks", 1))
        self.use_head_left = bool(model_cfg.get("use_head_left", False))
        support_camera_count = 3 if self.use_head_left else 2

        self.register_buffer("state_mean", torch.tensor(state_mean, dtype=torch.float32).view(1, state_dim), persistent=False)
        self.register_buffer("state_std", torch.tensor(state_std, dtype=torch.float32).view(1, state_dim), persistent=False)
        self.tactile_dim = tactile_dim
        self.tactile_out_dim = tactile_out
        if tactile_dim > 0:
            tactile_mean_values = tactile_mean if tactile_mean is not None else [0.0] * tactile_dim
            tactile_std_values = tactile_std if tactile_std is not None else [1.0] * tactile_dim
            self.register_buffer("tactile_mean", torch.tensor(tactile_mean_values, dtype=torch.float32).view(1, tactile_dim), persistent=False)
            self.register_buffer("tactile_std", torch.tensor(tactile_std_values, dtype=torch.float32).view(1, tactile_dim), persistent=False)

        self.wrist_resampler = WristResampler(
            dim=patch_dim,
            num_queries=resampler_queries,
            num_heads=resampler_heads,
            ff_hidden_dim=resampler_ff,
            num_blocks=resampler_blocks,
            dropout=dropout,
            num_cameras=support_camera_count,
        )
        self.ooi_wrist_blocks = nn.ModuleList(
            [CrossAttentionBlock(dim=patch_dim, num_heads=resampler_heads, ff_hidden_dim=resampler_ff, dropout=dropout) for _ in range(fusion_blocks)]
        )
        self.ooi_pooler = Pooler(dim=patch_dim, num_heads=resampler_heads, dropout=dropout)
        self.support_pooler = Pooler(dim=patch_dim, num_heads=resampler_heads, dropout=dropout)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_hidden),
            nn.GELU(),
            nn.Linear(state_hidden, state_out),
            nn.LayerNorm(state_out),
        )
        self.tactile_encoder = (
            nn.Sequential(
                nn.LayerNorm(tactile_dim),
                nn.Linear(tactile_dim, tactile_hidden),
                nn.GELU(),
                nn.Linear(tactile_hidden, tactile_out),
                nn.LayerNorm(tactile_out),
            )
            if tactile_dim > 0
            else None
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(patch_dim + state_out + tactile_out),
            nn.Linear(patch_dim + state_out + tactile_out, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, frame_dim),
            nn.LayerNorm(frame_dim),
        )

    def forward(
        self,
        ooi_tokens: torch.Tensor,
        head_left_tokens: torch.Tensor | None,
        wrist_left_tokens: torch.Tensor,
        wrist_right_tokens: torch.Tensor,
        state: torch.Tensor,
        tactile_60d: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        support_tokens = []
        camera_index = 0
        if self.use_head_left:
            if head_left_tokens is None:
                head_left_tokens = torch.zeros_like(wrist_left_tokens)
            support_tokens.append(self.wrist_resampler(head_left_tokens, camera_index=camera_index))
            camera_index += 1
        support_tokens.append(self.wrist_resampler(wrist_left_tokens, camera_index=camera_index))
        camera_index += 1
        support_tokens.append(self.wrist_resampler(wrist_right_tokens, camera_index=camera_index))
        support_context = torch.cat(support_tokens, dim=1)

        fused = ooi_tokens
        for block in self.ooi_wrist_blocks:
            fused = block(fused, support_context)

        posterior_visual = self.ooi_pooler(fused)
        prior_visual = self.support_pooler(support_context)
        if state.ndim == 2:
            state_mean = self.state_mean[:, : state.shape[-1]]
            state_std = self.state_std[:, : state.shape[-1]]
        elif state.ndim == 3:
            state_mean = self.state_mean[:, None, : state.shape[-1]]
            state_std = self.state_std[:, None, : state.shape[-1]]
        else:
            raise ValueError(f"Expected state to have rank 2 or 3, got shape {tuple(state.shape)}")
        state_norm = (state - state_mean) / state_std.clamp_min(1e-6)
        state_feature = self.state_encoder(state_norm)

        extra_features = [state_feature]
        if self.tactile_encoder is not None:
            if tactile_60d is None or tactile_60d.shape[-1] == 0:
                tactile_60d = state.new_zeros((*state.shape[:-1], self.tactile_dim))
            if tactile_60d.ndim == 2:
                tactile_mean = self.tactile_mean[:, : tactile_60d.shape[-1]]
                tactile_std = self.tactile_std[:, : tactile_60d.shape[-1]]
            elif tactile_60d.ndim == 3:
                tactile_mean = self.tactile_mean[:, None, : tactile_60d.shape[-1]]
                tactile_std = self.tactile_std[:, None, : tactile_60d.shape[-1]]
            else:
                raise ValueError(f"Expected tactile_60d to have rank 2 or 3, got shape {tuple(tactile_60d.shape)}")
            tactile_norm = (tactile_60d - tactile_mean) / tactile_std.clamp_min(1e-6)
            extra_features.append(self.tactile_encoder(tactile_norm))

        posterior = self.fusion(torch.cat([posterior_visual, *extra_features], dim=-1))
        prior = self.fusion(torch.cat([prior_visual, *extra_features], dim=-1))
        return {"posterior": posterior, "prior": prior}


@dataclass
class PlannerOutput:
    checkpoint_logits_raw: torch.Tensor
    raw_checkpoint_probs: torch.Tensor
    phase_logits: torch.Tensor
    prog_pred: torch.Tensor
    trans_pred: torch.Tensor
    raw_state_belief: torch.Tensor
    final_checkpoint_logits: torch.Tensor | None = None
    final_checkpoint_probs: torch.Tensor | None = None
    final_state_belief: torch.Tensor | None = None
    temporal_latent: torch.Tensor | None = None


class CheckpointPlannerModel(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        num_checkpoints: int,
        state_mean: list[float],
        state_std: list[float],
        tactile_mean: list[float] | None = None,
        tactile_std: list[float] | None = None,
    ) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.num_checkpoints = int(num_checkpoints)
        self.frame_dim = int(model_cfg.get("frame_dim", 512))
        temporal_heads = int(model_cfg.get("temporal_heads", 8))
        dropout = float(model_cfg.get("dropout", 0.1))
        rope_base = float(model_cfg.get("rope_base", 10000.0))
        temporal_layers = int(model_cfg.get("temporal_layers", 4))
        temporal_ffn = int(model_cfg.get("temporal_ffn_dim", 2048))
        nominal_dt = float(model_cfg.get("nominal_dt_seconds", 0.5))
        trunk_dim = int(model_cfg.get("trunk_dim", 256))

        self.nominal_dt_seconds = nominal_dt
        self.frame_encoder = FrameEncoder(
            config=config,
            state_mean=state_mean,
            state_std=state_std,
            tactile_mean=tactile_mean,
            tactile_std=tactile_std,
        )
        self.temporal = TemporalTransformer(
            dim=self.frame_dim,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            ff_hidden_dim=temporal_ffn,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.frame_dim),
            nn.Linear(self.frame_dim, trunk_dim),
            nn.GELU(),
            nn.LayerNorm(trunk_dim),
        )
        self.checkpoint_head = nn.Linear(trunk_dim, self.num_checkpoints + 1)
        self.phase_head = nn.Linear(trunk_dim, self.num_checkpoints * 2)
        self.progress_head = nn.Sequential(nn.Linear(trunk_dim, 2), nn.Sigmoid())

    def _positions_from_timestamps(self, timestamps: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.zeros_like(timestamps)
        for batch_index in range(timestamps.shape[0]):
            valid = valid_mask[batch_index]
            if bool(valid.any()):
                valid_times = timestamps[batch_index, valid]
                origin = valid_times[0]
                positions[batch_index, valid] = (valid_times - origin) / self.nominal_dt_seconds
        return positions

    def _decode_branch(
        self,
        frame_tokens: torch.Tensor,
        timestamps: torch.Tensor,
        valid_mask: torch.Tensor,
        batch: dict[str, torch.Tensor],
        apply_prior: bool,
        prior_config: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        batch_size = frame_tokens.shape[0]
        positions = self._positions_from_timestamps(timestamps=timestamps.float(), valid_mask=valid_mask)
        temporal = self.temporal(frame_tokens, positions=positions, valid_mask=valid_mask)
        current = temporal[:, -1, :]
        trunk = self.trunk(current)

        checkpoint_logits_raw = self.checkpoint_head(trunk)
        raw_checkpoint_probs = torch.softmax(checkpoint_logits_raw, dim=-1)
        phase_logits = self.phase_head(trunk).view(batch_size, self.num_checkpoints, 2)
        progress = self.progress_head(trunk)
        prog_pred = progress[:, 0]
        trans_pred = progress[:, 1]
        raw_state_belief = build_state_belief(raw_checkpoint_probs, phase_logits)

        outputs: dict[str, torch.Tensor] = {
            "checkpoint_logits_raw": checkpoint_logits_raw,
            "raw_checkpoint_probs": raw_checkpoint_probs,
            "phase_logits": phase_logits,
            "prog_pred": prog_pred,
            "trans_pred": trans_pred,
            "raw_state_belief": raw_state_belief,
            "temporal_latent": current,
        }

        if apply_prior and prior_config is not None and "last_completed_checkpoint" in batch:
            final_logits, final_probs = apply_continuity_prior(
                checkpoint_logits=checkpoint_logits_raw,
                last_completed_checkpoint=batch["last_completed_checkpoint"],
                gamma=float(prior_config.get("gamma", 10.0)),
                w_min=float(prior_config.get("w_min", 0.05)),
                w_future=float(prior_config.get("w_future", 0.01)),
                lambda_prior=float(prior_config.get("lambda_prior", 1.0)),
                eps=float(prior_config.get("eps", 1e-6)),
                gamma_future=(
                    None
                    if prior_config.get("gamma_future") in (None, "")
                    else float(prior_config.get("gamma_future", 15.0))
                ),
                current_ref_weight=float(prior_config.get("current_ref_weight", 0.3)),
                near_future_weight=float(prior_config.get("near_future_weight", 0.5)),
            )
            outputs["final_checkpoint_logits"] = final_logits
            outputs["final_checkpoint_probs"] = final_probs
            outputs["final_state_belief"] = build_state_belief(final_probs, phase_logits)
        else:
            # Keep the public branch-result schema complete when gamma
            # continuity is disabled.  In raw/no-continuity mode, "final" is
            # intentionally an alias of the raw prediction, not a reweighted
            # result. This does not alter either the posterior or prior model
            # branch computation.
            outputs["final_checkpoint_logits"] = checkpoint_logits_raw
            outputs["final_checkpoint_probs"] = raw_checkpoint_probs
            outputs["final_state_belief"] = raw_state_belief

        return outputs

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        apply_prior: bool = True,
        prior_config: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        ooi = batch["ooi_tokens"]
        head_left = batch.get("head_left_tokens")
        wl = batch["wrist_left_tokens"]
        wr = batch["wrist_right_tokens"]
        state = batch["state"]
        tactile_60d = batch.get("tactile_60d")
        timestamps = batch["timestamps"]
        valid_mask = batch["valid_mask"]

        batch_size, seq_len = ooi.shape[:2]
        flat_valid = valid_mask.view(-1)
        flat_count = batch_size * seq_len
        posterior_frame_tokens = ooi.new_zeros((flat_count, self.frame_dim), dtype=torch.float32)
        prior_frame_tokens = ooi.new_zeros((flat_count, self.frame_dim), dtype=torch.float32)

        if bool(flat_valid.any()):
            flat_ooi = ooi.view(flat_count, ooi.shape[-2], ooi.shape[-1])[flat_valid].float()
            flat_head_left = None
            if head_left is not None:
                flat_head_left = head_left.view(flat_count, head_left.shape[-2], head_left.shape[-1])[flat_valid].float()
            flat_wl = wl.view(flat_count, wl.shape[-2], wl.shape[-1])[flat_valid].float()
            flat_wr = wr.view(flat_count, wr.shape[-2], wr.shape[-1])[flat_valid].float()
            flat_state = state.view(flat_count, state.shape[-1])[flat_valid].float()
            flat_tactile = None
            if tactile_60d is not None and tactile_60d.shape[-1] > 0:
                flat_tactile = tactile_60d.view(flat_count, tactile_60d.shape[-1])[flat_valid].float()
            encoded = self.frame_encoder(flat_ooi, flat_head_left, flat_wl, flat_wr, flat_state, flat_tactile)
            posterior_frame_tokens[flat_valid] = encoded["posterior"]
            prior_frame_tokens[flat_valid] = encoded["prior"]

        posterior_frame_tokens = posterior_frame_tokens.view(batch_size, seq_len, self.frame_dim)
        prior_frame_tokens = prior_frame_tokens.view(batch_size, seq_len, self.frame_dim)
        posterior_outputs = self._decode_branch(
            posterior_frame_tokens,
            timestamps=timestamps,
            valid_mask=valid_mask,
            batch=batch,
            apply_prior=apply_prior,
            prior_config=prior_config,
        )
        prior_outputs = self._decode_branch(
            prior_frame_tokens,
            timestamps=timestamps,
            valid_mask=valid_mask,
            batch=batch,
            apply_prior=apply_prior,
            prior_config=prior_config,
        )

        outputs: dict[str, torch.Tensor] = dict(posterior_outputs)
        outputs["posterior"] = posterior_outputs
        outputs["prior"] = prior_outputs
        return outputs
