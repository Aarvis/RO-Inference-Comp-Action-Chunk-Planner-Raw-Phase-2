from __future__ import annotations

import torch


def gather_predictions(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    heatmap_prob = torch.sigmoid(outputs["heatmap_logits"])
    batch, _, grid_h, grid_w = heatmap_prob.shape
    flat_idx = heatmap_prob.flatten(2).argmax(dim=2).squeeze(1)
    gy = torch.div(flat_idx, grid_w, rounding_mode="floor")
    gx = flat_idx % grid_w
    offset = outputs["offset"]
    size = outputs["size"]
    pred = []
    for batch_index in range(batch):
        ox = offset[batch_index, 0, gy[batch_index], gx[batch_index]]
        oy = offset[batch_index, 1, gy[batch_index], gx[batch_index]]
        side = size[batch_index, 0, gy[batch_index], gx[batch_index]]
        cx = (gx[batch_index].float() + ox) / grid_w
        cy = (gy[batch_index].float() + oy) / grid_h
        conf = heatmap_prob[batch_index, 0, gy[batch_index], gx[batch_index]]
        pred.append(torch.stack([cx, cy, side, conf]))
    return torch.stack(pred, dim=0)
