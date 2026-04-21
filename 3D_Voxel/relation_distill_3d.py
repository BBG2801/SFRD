"""
3D inter-class relation distillation for SFRD/DDiF-style voxel codebases.

Design:
- frozen Conv3DNet.embed as the feature extractor
- compute class centroids on a balanced real mini-batch
- compute class centroids on decoded synthetic voxels
- match pairwise cosine relation matrices

This module is intentionally lightweight and independent from the main
distillation objective (DM / DC / TM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import Conv3DNet


Tensor = torch.Tensor


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class FrozenConv3DFeatureExtractor(nn.Module):
    """
    Frozen Conv3DNet.embed wrapper.

    By default it uses the same Conv3DNet definition already used by the 3D branch.
    """

    def __init__(
        self,
        channel: int,
        num_classes: int,
        im_size: Tuple[int, int, int],
        net_width: int = 128,
        net_depth: int = 3,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone = Conv3DNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            im_size=im_size,
        )
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.backbone.load_state_dict(state, strict=False)

        freeze_module(self.backbone)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone.embed(x)
        if feat.ndim != 2:
            feat = torch.flatten(feat, start_dim=1)
        return feat


def build_frozen_conv3d_extractor(
    channel: int,
    num_classes: int,
    im_size: Tuple[int, int, int],
    net_width: int = 128,
    net_depth: int = 3,
    checkpoint_path: Optional[str] = None,
) -> FrozenConv3DFeatureExtractor:
    return FrozenConv3DFeatureExtractor(
        channel=channel,
        num_classes=num_classes,
        im_size=im_size,
        net_width=net_width,
        net_depth=net_depth,
        checkpoint_path=checkpoint_path,
    )


def sample_balanced_real_batch(
    voxels_all: Tensor,
    indices_class: Sequence[Sequence[int]],
    batch_real_per_class: int,
    device: Union[str, torch.device],
    with_replacement: bool = False,
) -> Tuple[Tensor, Tensor]:
    xs: List[Tensor] = []
    ys: List[Tensor] = []

    for c, cls_indices in enumerate(indices_class):
        cls_indices = np.asarray(cls_indices)
        if len(cls_indices) == 0:
            raise ValueError(f"class {c} has no real samples")

        if with_replacement or len(cls_indices) < batch_real_per_class:
            chosen = np.random.choice(cls_indices, size=batch_real_per_class, replace=True)
        else:
            chosen = np.random.permutation(cls_indices)[:batch_real_per_class]

        x = voxels_all[chosen].to(device, non_blocking=True)
        y = torch.full((len(chosen),), c, dtype=torch.long, device=device)
        xs.append(x)
        ys.append(y)

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def compute_class_centroids(
    feats: Tensor,
    labels: Tensor,
    num_classes: int,
) -> Tensor:
    centroids = []
    for c in range(num_classes):
        mask = (labels == c)
        if not torch.any(mask):
            raise ValueError(f"Missing class {c} when computing centroids.")
        centroids.append(feats[mask].mean(dim=0))
    return torch.stack(centroids, dim=0)


def pairwise_cosine_relation_matrix(centroids: Tensor, eps: float = 1e-8) -> Tensor:
    centroids = F.normalize(centroids, dim=1, eps=eps)
    return centroids @ centroids.t()


@dataclass
class RelationStats:
    loss_rel: float
    centroid_mse: float
    real_rel_mean: float
    syn_rel_mean: float


class InterClassRelationDistillationLoss3D(nn.Module):
    """
    L_rel = ||R_syn - R_real||_F^2 / C^2
    where R is the pairwise cosine similarity matrix over class centroids.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        num_classes: int,
        eps: float = 1e-8,
        ignore_diag: bool = False,
    ) -> None:
        super().__init__()
        self.feature_extractor = freeze_module(feature_extractor)
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_diag = ignore_diag

    def _relation_loss_from_centroids(
        self,
        mu_real: Tensor,
        mu_syn: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        r_real = pairwise_cosine_relation_matrix(mu_real, eps=self.eps)
        r_syn = pairwise_cosine_relation_matrix(mu_syn, eps=self.eps)

        if self.ignore_diag:
            mask = ~torch.eye(self.num_classes, device=r_real.device, dtype=torch.bool)
            diff = r_syn[mask] - r_real[mask]
            loss_rel = (diff.pow(2).sum() / mask.sum().clamp_min(1))
        else:
            loss_rel = F.mse_loss(r_syn, r_real, reduction="sum") / float(self.num_classes * self.num_classes)

        details = {
            "R_real": r_real,
            "R_syn": r_syn,
            "mu_real": mu_real,
            "mu_syn": mu_syn,
            "centroid_mse": F.mse_loss(mu_syn, mu_real.detach()),
        }
        return loss_rel, details

    def forward_from_sfrd(
        self,
        synset,
        voxels_all: Tensor,
        indices_class: Sequence[Sequence[int]],
        batch_real_per_class: int,
        syn_decode_chunk: Optional[int] = None,
        return_details: bool = False,
        with_replacement: bool = False,
    ):
        device = next(self.feature_extractor.parameters()).device

        real_x, real_y = sample_balanced_real_batch(
            voxels_all=voxels_all,
            indices_class=indices_class,
            batch_real_per_class=batch_real_per_class,
            device=device,
            with_replacement=with_replacement,
        )
        with torch.no_grad():
            feat_real = self.feature_extractor(real_x)
            mu_real = compute_class_centroids(
                feat_real, real_y, num_classes=self.num_classes
            )

        label_syn = synset.label_syn
        mu_syn_list: List[Tensor] = []

        for c in range(self.num_classes):
            cls_idx = torch.nonzero(label_syn == c, as_tuple=False).view(-1).tolist()
            if len(cls_idx) == 0:
                raise ValueError(f"synthetic set has no samples for class {c}")

            if syn_decode_chunk is None or syn_decode_chunk <= 0:
                x_syn_c, _ = synset.get(indices=cls_idx)
                x_syn_c = x_syn_c.to(device)
                feat_syn_c = self.feature_extractor(x_syn_c)
                mu_syn_c = feat_syn_c.mean(dim=0)
            else:
                sum_feat = None
                count = 0
                for start in range(0, len(cls_idx), syn_decode_chunk):
                    sub_idx = cls_idx[start:start + syn_decode_chunk]
                    x_syn_sub, _ = synset.get(indices=sub_idx)
                    x_syn_sub = x_syn_sub.to(device)
                    feat_syn_sub = self.feature_extractor(x_syn_sub)
                    chunk_sum = feat_syn_sub.sum(dim=0)
                    sum_feat = chunk_sum if sum_feat is None else (sum_feat + chunk_sum)
                    count += feat_syn_sub.shape[0]
                mu_syn_c = sum_feat / float(count)

            mu_syn_list.append(mu_syn_c)

        mu_syn = torch.stack(mu_syn_list, dim=0)
        loss_rel, details = self._relation_loss_from_centroids(mu_real, mu_syn)
        return (loss_rel, details) if return_details else loss_rel


def build_relation_stats(details: Dict[str, Tensor], loss_rel: Tensor) -> RelationStats:
    return RelationStats(
        loss_rel=float(loss_rel.detach().item()),
        centroid_mse=float(details["centroid_mse"].detach().item()),
        real_rel_mean=float(details["R_real"].detach().mean().item()),
        syn_rel_mean=float(details["R_syn"].detach().mean().item()),
    )
