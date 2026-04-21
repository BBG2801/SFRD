"""
Inter-class Relation Distillation (IRD) module for DDiF-like codebases.

This module implements the relation loss described in the SFRD paper:
1) sample a balanced real mini-batch across classes
2) decode the synthetic set
3) extract frozen features
4) compute per-class centroids
5) build pairwise cosine relation matrices
6) minimize Frobenius distance between real/synthetic relation matrices

Designed to be plug-and-play with DDiF-style projects where:
- real data are stored as `images_all` and `indices_class`
- synthetic samples are obtained by `synset.get(indices=None, need_copy=False)`

Author: ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze all parameters and switch to eval mode."""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def build_input_preprocess(
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    clamp: bool = True,
    repeat_gray_to_rgb: bool = False,
    resize_to: Optional[Tuple[int, int]] = None,
    imagenet_norm: bool = False,
) -> Callable[[Tensor], Tensor]:
    """
    Build a lightweight preprocessing function for frozen feature extractors.

    Typical image use:
        preprocess = build_input_preprocess(
            mean=args.mean, std=args.std,
            clamp=True, repeat_gray_to_rgb=False,
            resize_to=(224, 224), imagenet_norm=True
        )
    """
    mean_t = None
    std_t = None
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(1, -1, 1, 1)
        std_t = torch.tensor(std).view(1, -1, 1, 1)

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _preprocess(x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [N,C,H,W], got shape={tuple(x.shape)}")

        out = x
        if mean_t is not None and std_t is not None:
            out = out * std_t.to(out.device, out.dtype) + mean_t.to(out.device, out.dtype)

        if clamp:
            out = out.clamp(0.0, 1.0)

        if repeat_gray_to_rgb and out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)

        if resize_to is not None:
            out = F.interpolate(out, size=resize_to, mode="bilinear", align_corners=False)

        if imagenet_norm:
            if out.shape[1] != 3:
                raise ValueError(
                    f"ImageNet normalization requires 3 channels, got C={out.shape[1]}"
                )
            out = (out - imagenet_mean.to(out.device, out.dtype)) / imagenet_std.to(out.device, out.dtype)

        return out

    return _preprocess


class FeatureExtractorWrapper(nn.Module):
    """
    Wrap a frozen backbone and always return a 2D feature tensor [N, D].

    You can either:
    - pass a backbone whose forward already returns features, or
    - pass a feature_fn(backbone, x) -> features
    """

    def __init__(
        self,
        backbone: nn.Module,
        preprocess: Optional[Callable[[Tensor], Tensor]] = None,
        feature_fn: Optional[Callable[[nn.Module, Tensor], Tensor]] = None,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = freeze_module(backbone)
        self.preprocess = preprocess
        self.feature_fn = feature_fn
        self.flatten = flatten

    def _default_extract(self, x: Tensor) -> Tensor:
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone(x)

        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if self.flatten and feat.ndim > 2:
            feat = torch.flatten(feat, start_dim=1)

        if feat.ndim != 2:
            raise ValueError(
                f"Feature extractor must return [N,D] after wrapping, got {tuple(feat.shape)}"
            )
        return feat

    def forward(self, x: Tensor) -> Tensor:
        if self.preprocess is not None:
            x = self.preprocess(x)
        if self.feature_fn is not None:
            feat = self.feature_fn(self.backbone, x)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if self.flatten and feat.ndim > 2:
                feat = torch.flatten(feat, start_dim=1)
            return feat
        return self._default_extract(x)


def build_frozen_resnet18_extractor(
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    resize_to: Tuple[int, int] = (224, 224),
    imagenet_weights: bool = True,
) -> FeatureExtractorWrapper:
    """
    A practical default for image-domain experiments.
    This keeps only the convolutional trunk + global pooling and freezes it.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if imagenet_weights else None
    net = resnet18(weights=weights)
    trunk = nn.Sequential(*list(net.children())[:-1])  # [N, 512, 1, 1]

    preprocess = build_input_preprocess(
        mean=mean,
        std=std,
        clamp=True,
        repeat_gray_to_rgb=False,
        resize_to=resize_to,
        imagenet_norm=imagenet_weights,
    )
    return FeatureExtractorWrapper(trunk, preprocess=preprocess, feature_fn=None, flatten=True)


def sample_balanced_real_batch(
    images_all: Tensor,
    indices_class: Sequence[Sequence[int]],
    batch_real_per_class: int,
    device: Union[str, torch.device],
    with_replacement: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    DDiF-style balanced sampling from cached real data.

    Args:
        images_all: [N, C, H, W] on CPU or GPU
        indices_class: list of per-class index lists
        batch_real_per_class: sampled real images for each class
        device: output device
    """
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

        x = images_all[chosen].to(device, non_blocking=True)
        y = torch.full((len(chosen),), c, dtype=torch.long, device=device)
        xs.append(x)
        ys.append(y)

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _compute_centroids_hard(
    feats: Tensor,
    labels: Tensor,
    num_classes: int,
    eps: float,
) -> Tensor:
    centroids = []
    for c in range(num_classes):
        mask = (labels == c)
        if not torch.any(mask):
            raise ValueError(f"Missing class {c} when computing centroids.")
        centroids.append(feats[mask].mean(dim=0))
    return torch.stack(centroids, dim=0)


def _compute_centroids_soft(
    feats: Tensor,
    soft_labels: Tensor,
    eps: float,
) -> Tensor:
    # soft_labels: [N, C]
    weights = soft_labels / soft_labels.sum(dim=0, keepdim=True).clamp_min(eps)
    # centroids[c] = sum_i w_ic * feat_i
    return weights.transpose(0, 1) @ feats


def compute_class_centroids(
    feats: Tensor,
    labels: Tensor,
    num_classes: Optional[int] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Supports:
    - hard labels: [N]
    - soft labels: [N, C]
    """
    if labels.ndim == 1:
        if num_classes is None:
            num_classes = int(labels.max().item()) + 1
        return _compute_centroids_hard(feats, labels, num_classes, eps)

    if labels.ndim == 2:
        return _compute_centroids_soft(feats, labels, eps)

    raise ValueError(f"Unsupported label shape: {tuple(labels.shape)}")


def pairwise_cosine_relation_matrix(centroids: Tensor, eps: float = 1e-8) -> Tensor:
    centroids = F.normalize(centroids, dim=1, eps=eps)
    return centroids @ centroids.t()


@dataclass
class RelationStats:
    loss_rel: float
    centroid_mse: float
    real_rel_mean: float
    syn_rel_mean: float


class InterClassRelationDistillationLoss(nn.Module):
    """
    Implements:
        mu_real_c = mean_{x in B_c} g_omega(x)
        mu_syn_c  = mean_{k} g_omega(x_syn_{c,k})
        R = cosine(mu_c, mu_c')
        L_rel = ||R_syn - R_real||_F^2 / C^2
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

    def forward(
        self,
        real_x: Tensor,
        real_y: Tensor,
        syn_x: Tensor,
        syn_y: Tensor,
        return_details: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        # Real branch can be detached to save memory.
        with torch.no_grad():
            feat_real = self.feature_extractor(real_x)
            mu_real = compute_class_centroids(
                feat_real, real_y, num_classes=self.num_classes, eps=self.eps
            )

        # Synthetic branch must keep gradient wrt syn_x.
        feat_syn = self.feature_extractor(syn_x)
        mu_syn = compute_class_centroids(
            feat_syn, syn_y, num_classes=self.num_classes, eps=self.eps
        )

        loss_rel, details = self._relation_loss_from_centroids(mu_real, mu_syn)
        return (loss_rel, details) if return_details else loss_rel

    def forward_from_ddif(
        self,
        synset,
        images_all: Tensor,
        indices_class: Sequence[Sequence[int]],
        batch_real_per_class: int,
        syn_decode_chunk: Optional[int] = None,
        return_details: bool = False,
        with_replacement: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Memory-friendly DDiF-style version.

        It:
        1) samples balanced real images from `images_all` + `indices_class`
        2) decodes synthetic samples from `synset.get(...)`
        3) computes class centroids
        4) returns relation loss

        Notes:
        - `synset.label_syn` in DDiF is hard labels arranged class-by-class.
        - if `syn_decode_chunk` is None, one class is decoded at once.
        """
        device = next(self.feature_extractor.parameters()).device

        # ---- real centroids ----
        real_x, real_y = sample_balanced_real_batch(
            images_all=images_all,
            indices_class=indices_class,
            batch_real_per_class=batch_real_per_class,
            device=device,
            with_replacement=with_replacement,
        )
        with torch.no_grad():
            feat_real = self.feature_extractor(real_x)
            mu_real = compute_class_centroids(
                feat_real, real_y, num_classes=self.num_classes, eps=self.eps
            )

        # ---- synthetic centroids ----
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
