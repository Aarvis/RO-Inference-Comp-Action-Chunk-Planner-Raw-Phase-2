from __future__ import annotations

import torch


class PlannerHistoryBuffer:
    def __init__(
        self,
        *,
        sequence_length: int,
        patch_count: int,
        patch_dim: int,
        state_dim: int,
        tactile_dim: int,
        device: torch.device,
    ) -> None:
        self.sequence_length = int(sequence_length)
        self.patch_count = int(patch_count)
        self.patch_dim = int(patch_dim)
        self.state_dim = int(state_dim)
        self.tactile_dim = int(tactile_dim)
        self.device = device

        token_shape = (self.sequence_length, self.patch_count, self.patch_dim)
        self.head_left_tokens = torch.zeros(token_shape, dtype=torch.float16, device=device)
        self.wrist_left_tokens = torch.zeros_like(self.head_left_tokens)
        self.wrist_right_tokens = torch.zeros_like(self.head_left_tokens)
        self.ooi_tokens = torch.zeros_like(self.head_left_tokens)
        self.state = torch.zeros((self.sequence_length, self.state_dim), dtype=torch.float32, device=device)
        self.tactile_60d = torch.zeros((self.sequence_length, self.tactile_dim), dtype=torch.float32, device=device)
        self.timestamps = torch.zeros((self.sequence_length,), dtype=torch.float32, device=device)
        self.ooi_present = torch.zeros((self.sequence_length,), dtype=torch.bool, device=device)
        self.count = 0
        self.next_index = 0

    def reset(self) -> None:
        self.head_left_tokens.zero_()
        self.wrist_left_tokens.zero_()
        self.wrist_right_tokens.zero_()
        self.ooi_tokens.zero_()
        self.state.zero_()
        self.tactile_60d.zero_()
        self.timestamps.zero_()
        self.ooi_present.zero_()
        self.count = 0
        self.next_index = 0

    @property
    def num_valid_steps(self) -> int:
        return int(self.count)

    def append(
        self,
        *,
        head_left_tokens: torch.Tensor,
        wrist_left_tokens: torch.Tensor,
        wrist_right_tokens: torch.Tensor,
        ooi_tokens: torch.Tensor,
        state: torch.Tensor,
        tactile_60d: torch.Tensor,
        timestamp: float,
        ooi_valid: bool = True,
    ) -> None:
        slot = int(self.next_index)
        self.head_left_tokens[slot].copy_(self._coerce_tokens(head_left_tokens))
        self.wrist_left_tokens[slot].copy_(self._coerce_tokens(wrist_left_tokens))
        self.wrist_right_tokens[slot].copy_(self._coerce_tokens(wrist_right_tokens))
        self.ooi_tokens[slot].copy_(self._coerce_tokens(ooi_tokens))
        self.state[slot].copy_(state.to(device=self.device, dtype=torch.float32))
        self.tactile_60d[slot].copy_(tactile_60d.to(device=self.device, dtype=torch.float32))
        self.timestamps[slot] = float(timestamp)
        self.ooi_present[slot] = bool(ooi_valid)

        if self.count < self.sequence_length:
            self.count += 1
        self.next_index = (self.next_index + 1) % self.sequence_length

    def _coerce_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 3 and tokens.shape[0] == 1:
            tokens = tokens[0]
        if tokens.shape != (self.patch_count, self.patch_dim):
            raise ValueError(
                f"Expected tokens with shape {(self.patch_count, self.patch_dim)}, got {tuple(tokens.shape)}."
            )
        return tokens.to(device=self.device, dtype=torch.float16)

    def _ordered_indices(self) -> torch.Tensor:
        if self.count <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        if self.count < self.sequence_length:
            return torch.arange(self.count, device=self.device, dtype=torch.long)
        return (torch.arange(self.sequence_length, device=self.device, dtype=torch.long) + self.next_index) % self.sequence_length

    def build_batch(self, *, last_completed_checkpoint: int) -> dict[str, torch.Tensor]:
        batch_head_left = torch.zeros_like(self.head_left_tokens)
        batch_wl = torch.zeros_like(self.wrist_left_tokens)
        batch_wr = torch.zeros_like(self.wrist_right_tokens)
        batch_ooi = torch.zeros_like(self.ooi_tokens)
        batch_state = torch.zeros_like(self.state)
        batch_tactile = torch.zeros_like(self.tactile_60d)
        batch_timestamps = torch.zeros_like(self.timestamps)
        valid_mask = torch.zeros((self.sequence_length,), dtype=torch.bool, device=self.device)
        ooi_valid_mask = torch.zeros((self.sequence_length,), dtype=torch.bool, device=self.device)

        if self.count > 0:
            ordered = self._ordered_indices()
            start = self.sequence_length - self.count
            batch_head_left[start:] = self.head_left_tokens.index_select(0, ordered)
            batch_wl[start:] = self.wrist_left_tokens.index_select(0, ordered)
            batch_wr[start:] = self.wrist_right_tokens.index_select(0, ordered)
            batch_ooi[start:] = self.ooi_tokens.index_select(0, ordered)
            batch_state[start:] = self.state.index_select(0, ordered)
            batch_tactile[start:] = self.tactile_60d.index_select(0, ordered)
            batch_timestamps[start:] = self.timestamps.index_select(0, ordered)
            valid_mask[start:] = True
            ooi_valid_mask[start:] = self.ooi_present.index_select(0, ordered)

        return {
            "head_left_tokens": batch_head_left.unsqueeze(0),
            "wrist_left_tokens": batch_wl.unsqueeze(0),
            "wrist_right_tokens": batch_wr.unsqueeze(0),
            "ooi_tokens": batch_ooi.unsqueeze(0),
            "state": batch_state.unsqueeze(0),
            "tactile_60d": batch_tactile.unsqueeze(0),
            "timestamps": batch_timestamps.unsqueeze(0),
            "valid_mask": valid_mask.unsqueeze(0),
            "ooi_valid_mask": ooi_valid_mask.unsqueeze(0),
            "last_completed_checkpoint": torch.tensor([int(last_completed_checkpoint)], dtype=torch.long, device=self.device),
        }
