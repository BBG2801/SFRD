import os
import math
import time
import copy
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from math import sqrt
from utils import save_and_print


def to_coordinates_and_features(volume):
    """
    Accept 3D volume in one of:
      [C, D, H, W]
      [D, C, H, W]
      [D, H, W, C]

    Return:
      coordinates: [D*H*W, 3] for (z, y, x), normalized to [-1, 1]
      features:    [D*H*W, C]
      volume_cdhw: [C, D, H, W]
    """
    if volume is None:
        raise ValueError("to_coordinates_and_features got None volume")

    if not torch.is_tensor(volume):
        volume = torch.tensor(volume)

    volume = volume.float()

    if volume.dim() != 4:
        raise ValueError(f"Expected 4D volume tensor, got shape {tuple(volume.shape)}")

    if volume.shape[0] in (1, 3):
        volume_cdhw = volume.contiguous()
    elif volume.shape[1] in (1, 3):
        volume_cdhw = volume.permute(1, 0, 2, 3).contiguous()
    elif volume.shape[-1] in (1, 3):
        volume_cdhw = volume.permute(3, 0, 1, 2).contiguous()
    else:
        raise ValueError(
            f"Cannot infer channel dimension from volume shape {tuple(volume.shape)}. "
            f"Expected one of [C,D,H,W], [D,C,H,W], [D,H,W,C]."
        )

    c, d, h, w = volume_cdhw.shape

    zs = torch.linspace(-1.0, 1.0, steps=d, device=volume_cdhw.device)
    ys = torch.linspace(-1.0, 1.0, steps=h, device=volume_cdhw.device)
    xs = torch.linspace(-1.0, 1.0, steps=w, device=volume_cdhw.device)

    grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
    coordinates = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(-1, 3)
    features = volume_cdhw.permute(1, 2, 3, 0).reshape(-1, c).contiguous()

    return coordinates, features, volume_cdhw


def _voxel_to_xyz(voxel, threshold=0.5):
    """
    voxel: [C, D, H, W] or [D, H, W]
    return xs, ys, zs of occupied cells
    """
    if torch.is_tensor(voxel):
        voxel = voxel.detach().cpu()

    if voxel.dim() == 4:
        voxel = voxel[0]  # [D, H, W]
    elif voxel.dim() != 3:
        raise ValueError(f"Expected voxel with 3 or 4 dims, got {tuple(voxel.shape)}")

    occ = voxel > threshold
    coords = occ.nonzero(as_tuple=False)

    if coords.numel() == 0:
        return np.array([]), np.array([]), np.array([])

    zs = coords[:, 0].numpy()
    ys = coords[:, 1].numpy()
    xs = coords[:, 2].numpy()
    return xs, ys, zs


def save_voxel_grid_3d(
    voxels,
    save_path,
    labels=None,
    ncols=2,
    threshold=0.5,
    max_items=10,
    elev=24,
    azim=-62,
    point_size=10,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if torch.is_tensor(voxels):
        voxels = voxels.detach().cpu()

    n = min(int(voxels.shape[0]), int(max_items))
    if n < 1:
        raise ValueError("No voxels to visualize.")

    ncols = max(1, min(int(ncols), n))
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        xs, ys, zs = _voxel_to_xyz(voxels[i], threshold=threshold)

        if len(xs) > 0:
            ax.scatter(
                xs, ys, zs,
                s=point_size,
                c="#8B0000",
                marker="s",
                edgecolors="k",
                linewidths=0.15,
            )

        if voxels[i].dim() == 4:
            _, d, h, w = voxels[i].shape
        else:
            d, h, w = voxels[i].shape

        ax.set_xlim(0, w - 1)
        ax.set_ylim(0, h - 1)
        ax.set_zlim(0, d - 1)
        ax.set_box_aspect((w, h, d))
        ax.view_init(elev=elev, azim=azim)

        if labels is not None:
            ax.set_title(f"cls={int(labels[i])}", fontsize=10)
        else:
            ax.set_title(f"id={i}", fontsize=10)

        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_voxel_single_3d(
    voxel,
    save_path,
    label=None,
    threshold=0.5,
    elev=24,
    azim=-62,
    point_size=10,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if torch.is_tensor(voxel):
        voxel = voxel.detach().cpu()

    fig = plt.figure(figsize=(5.5, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = _voxel_to_xyz(voxel, threshold=threshold)
    if len(xs) > 0:
        ax.scatter(
            xs, ys, zs,
            s=point_size,
            c="#8B0000",
            marker="s",
            edgecolors="k",
            linewidths=0.15,
        )

    if voxel.dim() == 4:
        _, d, h, w = voxel.shape
    else:
        d, h, w = voxel.shape

    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)
    ax.set_zlim(0, d - 1)
    ax.set_box_aspect((w, h, d))
    ax.view_init(elev=elev, azim=azim)

    if label is not None:
        ax.set_title(f"cls={int(label)}", fontsize=10)

    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


class SFRD:
    """
    Per-class shared backbone + per-instance shift modulation for 3D voxel data.
    Internal decoder outputs logits; get() returns sigmoid probabilities in [0,1].
    """
    def __init__(self, args):
        self.args = args
        self.log_path = args.log_path
        self.channel = args.channel
        self.num_classes = args.num_classes
        self.im_size = args.im_size
        self.device = args.device
        self.ipc = args.ipc

        if len(self.im_size) != 3:
            raise ValueError(f"Expected 3D im_size, got {self.im_size}")

        self.depth, self.height, self.width = self.im_size

        self.dim_in = args.dim_in
        self.dim_out = args.dim_out
        self.w0_initial = args.w0_initial
        self.w0 = args.w0

        if self.dim_in != 3:
            raise ValueError(
                f"For voxel distillation, dim_in must be 3 for (z,y,x), but got dim_in={self.dim_in}"
            )
        if self.dim_out != self.channel:
            raise ValueError(
                f"dim_out must equal channel for voxel reconstruction, but got dim_out={self.dim_out}, channel={self.channel}"
            )

        self.num_layers = args.num_layers
        self.layer_size = args.layer_size

        self.lr_nf = args.lr_nf
        self.lr_nf_init = args.lr_nf_init

        self.shared_mode = getattr(args, "shared_mode", "per_class")
        self.modulation_type = getattr(args, "modulation_type", "shift")
        if self.modulation_type != "shift":
            raise ValueError(f"Only shift modulation is supported, but got modulation_type={self.modulation_type}")

        self.shift_init = getattr(args, "shift_init", 0.0)
        self.latent_std = getattr(args, "latent_std", 0.01)

        # More conservative learning-rate handling.
        raw_lr_bkb = getattr(args, "lr_nf_backbone", None)
        raw_lr_shift = getattr(args, "lr_nf_shift", None)
        raw_lr_init_bkb = getattr(args, "lr_nf_init_backbone", None)
        raw_lr_init_shift = getattr(args, "lr_nf_init_shift", None)

        self.lr_nf_backbone = self.lr_nf if raw_lr_bkb is None else raw_lr_bkb
        self.lr_nf_shift = self.lr_nf if raw_lr_shift is None else min(raw_lr_shift, self.lr_nf * 2)

        self.lr_nf_init_backbone = self.lr_nf_init if raw_lr_init_bkb is None else raw_lr_init_bkb
        self.lr_nf_init_shift = self.lr_nf_init_backbone if raw_lr_init_shift is None else min(raw_lr_init_shift, self.lr_nf_init_backbone)

        self.epochs_init = getattr(self.args, "epochs_init", 5000)
        self.init_batch_per_step = getattr(self.args, "init_batch_per_step", 8)
        self.init_instances_per_epoch = getattr(self.args, "init_instances_per_epoch", -1)

        self.train_backbone = getattr(args, "train_backbone", True)
        self.train_latent = getattr(args, "train_latent", True)

        if (not self.train_backbone) and (not self.train_latent):
            raise ValueError("At least one of train_backbone or train_latent must be True.")

        # Init regularization
        self.init_pos_weight_cap = float(getattr(args, "init_pos_weight_cap", 16.0))
        self.init_lambda_occ = float(getattr(args, "init_lambda_occ", 0.2))
        self.init_lambda_tv = float(getattr(args, "init_lambda_tv", 0.01))
        self.init_patience = int(getattr(args, "init_patience", 600))

        self.shift_dim_per_instance = self.num_layers * self.layer_size
        self.latent_dim = self.shift_dim_per_instance

        backbone_temp = SharedModulatedSiren(
            dim_in=self.dim_in,
            dim_hidden=self.layer_size,
            dim_out=self.dim_out,
            num_layers=self.num_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=self.w0_initial,
            w0=self.w0,
        )

        self.backbone_params_per_class = sum(
            sum(t.nelement() for t in tensors)
            for tensors in (backbone_temp.parameters(), backbone_temp.buffers())
        )
        self.backbone_params_total = self.num_classes * self.backbone_params_per_class
        self.latent_params_per_instance = self.latent_dim

        allowed_total_budget = (
            self.num_classes * self.ipc * self.channel * self.depth * self.height * self.width
        )

        if self.args.dipc > 0:
            self.num_per_class = self.args.dipc
        else:
            total_num_instances = int(
                (allowed_total_budget - self.backbone_params_total) / self.latent_params_per_instance
            )
            self.num_per_class = total_num_instances // self.num_classes

        self.total_num_instances = self.num_classes * self.num_per_class
        self.total_budget = self.backbone_params_total + self.total_num_instances * self.latent_params_per_instance

        if (self.total_budget > allowed_total_budget) or (self.num_per_class < 1):
            msg = (
                f"Invalid Budget: total_budget={self.total_budget}, "
                f"allowed_total_budget={allowed_total_budget}, "
                f"num_per_class={self.num_per_class}, "
                f"backbone_params_per_class={self.backbone_params_per_class}, "
                f"backbone_params_total={self.backbone_params_total}, "
                f"latent_params_per_instance={self.latent_params_per_instance}"
            )
            save_and_print(self.log_path, msg)
            raise ValueError(msg)

        self.init_loss_tag = (
            f"bcecap{self.init_pos_weight_cap:g}"
            f"_occ{self.init_lambda_occ:g}"
            f"_tv{self.init_lambda_tv:g}"
            f"_clipshift"
        )

        del backbone_temp

    def _build_single_backbone(self):
        return SharedModulatedSiren(
            dim_in=self.dim_in,
            dim_hidden=self.layer_size,
            dim_out=self.dim_out,
            num_layers=self.num_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=self.w0_initial,
            w0=self.w0,
        ).to(self.device)

    def _apply_train_mode(self):
        for p in self.nf_syn.parameters():
            p.requires_grad_(self.train_backbone)
        self.shift_syn.requires_grad_(self.train_latent)

        for backbone in self.nf_syn:
            if hasattr(backbone, "use_checkpoint"):
                backbone.use_checkpoint = bool(self.train_backbone)

    def _build_optimizer(self, for_init=False):
        param_groups = []

        lr_backbone = self.lr_nf_init_backbone if for_init else self.lr_nf_backbone
        lr_shift = self.lr_nf_init_shift if for_init else self.lr_nf_shift

        if self.train_backbone:
            backbone_params = [p for p in self.nf_syn.parameters() if p.requires_grad]
            if len(backbone_params) > 0:
                param_groups.append({"params": backbone_params, "lr": lr_backbone})

        if self.train_latent and self.shift_syn.requires_grad:
            param_groups.append({"params": [self.shift_syn], "lr": lr_shift})

        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found when building optimizer.")

        return torch.optim.Adam(param_groups)

    def _forward_instance(self, class_id, coord, shift, detach_backbone=False):
        backbone = self.nf_syn[class_id]
        if detach_backbone:
            with torch.no_grad():
                return backbone(coord, shift=shift)
        return backbone(coord, shift=shift)

    def _export_backbones_cpu(self):
        return [
            {k: copy.deepcopy(v.to("cpu")) for k, v in backbone.state_dict().items()}
            for backbone in self.nf_syn
        ]

    def _estimate_pos_weight(self, volumes_real):
        with torch.no_grad():
            target = volumes_real.float().clamp(0.0, 1.0)
            pos = float(target.sum().item())
            total = float(target.numel())
            neg = max(total - pos, 1.0)
            if pos < 1.0:
                return 1.0
            pos_weight = neg / pos
            pos_weight = max(1.0, min(pos_weight, self.init_pos_weight_cap))
        return float(pos_weight)

    def _tv3d(self, voxel_5d):
        # voxel_5d: [B,C,D,H,W]
        tv = 0.0
        tv = tv + (voxel_5d[:, :, 1:, :, :] - voxel_5d[:, :, :-1, :, :]).abs().mean()
        tv = tv + (voxel_5d[:, :, :, 1:, :] - voxel_5d[:, :, :, :-1, :]).abs().mean()
        tv = tv + (voxel_5d[:, :, :, :, 1:] - voxel_5d[:, :, :, :, :-1]).abs().mean()
        return tv

    def _build_initialized_synset_path(self):
        dipc = getattr(self.args, "dipc", 0)
        subset = getattr(self.args, "subset", "none")
        res = getattr(self.args, "res", f"{self.depth}x{self.height}x{self.width}")

        initialized_synset_path = (
            f"../initialized_synset/"
            f"{self.args.dataset}_{subset}_{res}_{self.args.model}_{self.args.ipc}ipc_{dipc}dipc/"
            f"init_perclasssharedshift#({self.dim_in},{self.num_layers},{self.layer_size},{self.dim_out})"
            f"_shift{self.shift_dim_per_instance}"
            f"_({self.w0_initial:g},{self.w0:g})"
            f"_({self.epochs_init},bkb{self.lr_nf_init_backbone:.0e},sh{self.lr_nf_init_shift:.0e})"
            f"_{self.init_loss_tag}"
            f"_inst{self.init_instances_per_epoch}_bs{self.init_batch_per_step}.pt"
        )
        return initialized_synset_path

    def init(self, volumes_real, labels_real, indices_class):
        save_and_print(self.log_path, "=" * 50 + "\n SynSet Initialization")

        pos_weight_value = self._estimate_pos_weight(volumes_real)
        save_and_print(self.log_path, f"[Init] BCE pos_weight = {pos_weight_value:.4f}")
        save_and_print(
            self.log_path,
            f"[Init] lambda_occ = {self.init_lambda_occ}, lambda_tv = {self.init_lambda_tv}, "
            f"lr_init_backbone = {self.lr_nf_init_backbone}, lr_init_shift = {self.lr_nf_init_shift}"
        )

        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], device=self.device)
        ).to(self.device)

        volume_temp = torch.rand((self.channel, self.depth, self.height, self.width), device=self.device)
        self.coord, _, _ = to_coordinates_and_features(volume_temp)
        self.coord = self.coord.to(self.device)
        del volume_temp

        self.nf_syn = nn.ModuleList([self._build_single_backbone() for _ in range(self.num_classes)])

        self.shift_syn = nn.Parameter(
            torch.randn(
                self.num_classes,
                self.num_per_class,
                self.shift_dim_per_instance,
                device=self.device,
            ) * self.latent_std + self.shift_init
        )

        self._apply_train_mode()
        initialized_synset_path = self._build_initialized_synset_path()

        if os.path.isfile(initialized_synset_path):
            save_and_print(self.log_path, f"\n Load from >>>>> {initialized_synset_path} \n")
            data = torch.load(initialized_synset_path, map_location="cpu")

            if "shared_nf_per_class" not in data or "shift" not in data:
                raise ValueError(f"Incompatible synset format: {initialized_synset_path}")

            nf_states = data["shared_nf_per_class"]
            if len(nf_states) != self.num_classes:
                raise ValueError(
                    f"shared_nf_per_class length mismatch: got {len(nf_states)}, expected {self.num_classes}"
                )

            for c in range(self.num_classes):
                self.nf_syn[c].load_state_dict(nf_states[c])

            shift_state = data["shift"]
            expected_shape = (self.num_classes, self.num_per_class, self.shift_dim_per_instance)
            if tuple(shift_state.shape) != expected_shape:
                raise ValueError(
                    f"shift shape mismatch: got {tuple(shift_state.shape)}, expected {expected_shape}"
                )

            self.shift_syn.data.copy_(shift_state.to(self.device))

        else:
            save_and_print(self.log_path, f"\n No initialized synset >>>>> {initialized_synset_path} \n")

            volumes_init = []
            class_ids = []
            instance_ids = []

            for c in range(self.num_classes):
                cls_indices = np.array(indices_class[c])
                if len(cls_indices) < self.num_per_class:
                    raise ValueError(
                        f"Class {c} has only {len(cls_indices)} real samples, "
                        f"but num_per_class={self.num_per_class}."
                    )

                sampled_indices = np.random.permutation(cls_indices)[:self.num_per_class]
                sampled_volumes = volumes_real[sampled_indices]
                volumes_init.append(sampled_volumes)
                class_ids.extend([c] * self.num_per_class)
                instance_ids.extend(list(range(self.num_per_class)))

            volumes_init = torch.cat(volumes_init, dim=0)
            class_ids = torch.tensor(class_ids, dtype=torch.long, device=self.device)
            instance_ids = torch.tensor(instance_ids, dtype=torch.long, device=self.device)

            init_optimizer = self._build_optimizer(for_init=True)

            total_instances = self.total_num_instances
            if self.init_instances_per_epoch is None or self.init_instances_per_epoch <= 0:
                instances_per_epoch = total_instances
            else:
                instances_per_epoch = min(self.init_instances_per_epoch, total_instances)

            save_and_print(
                self.log_path,
                f"Initialization uses {total_instances} total instances, "
                f"{instances_per_epoch} instances per epoch, "
                f"batch size {self.init_batch_per_step}, "
                f"epochs {self.epochs_init}."
            )

            best_init_loss = float("inf")
            best_nf_state = None
            best_shift_state = None
            bad_rounds = 0
            epoch_loss_list = []

            for ep in tqdm(range(self.epochs_init), desc="Init epochs"):
                perm_all = torch.randperm(total_instances, device=self.device)
                active_indices = perm_all[:instances_per_epoch]
                active_indices = active_indices[torch.randperm(active_indices.numel(), device=self.device)]

                batch_losses = []

                for start in range(0, active_indices.numel(), self.init_batch_per_step):
                    batch_flat_idx = active_indices[start:start + self.init_batch_per_step]

                    init_optimizer.zero_grad(set_to_none=True)
                    batch_loss = 0.0

                    for flat_idx in batch_flat_idx.tolist():
                        volume = volumes_init[flat_idx].to(self.device)
                        vol_coord, vol_value, vol_cdhw = to_coordinates_and_features(volume)
                        vol_target = vol_value.clamp(0.0, 1.0)

                        c = class_ids[flat_idx].item()
                        k = instance_ids[flat_idx].item()

                        shift = self.shift_syn[c, k]
                        pred_logits = self._forward_instance(c, vol_coord, shift=shift, detach_backbone=False)

                        if pred_logits.shape != vol_target.shape:
                            raise RuntimeError(
                                f"Shape mismatch before BCE: predicted={tuple(pred_logits.shape)}, "
                                f"target={tuple(vol_target.shape)}, "
                                f"volume_canonical_shape={tuple(vol_cdhw.shape)}"
                            )

                        pred_prob = torch.sigmoid(pred_logits)

                        bce_loss = criterion(pred_logits, vol_target)

                        target_occ = vol_target.mean()
                        occ_loss = (pred_prob.mean() - target_occ).pow(2)

                        pred_prob_5d = pred_prob.reshape(
                            self.depth, self.height, self.width, self.channel
                        ).permute(3, 0, 1, 2).unsqueeze(0).contiguous()

                        tv_loss = self._tv3d(pred_prob_5d)

                        loss = bce_loss + self.init_lambda_occ * occ_loss + self.init_lambda_tv * tv_loss
                        batch_loss = batch_loss + loss

                    batch_loss = batch_loss / batch_flat_idx.numel()
                    batch_loss.backward()
                    init_optimizer.step()
                    batch_losses.append(batch_loss.item())

                mean_epoch_loss = float(np.mean(batch_losses)) if len(batch_losses) > 0 else 0.0
                epoch_loss_list.append(mean_epoch_loss)

                if mean_epoch_loss < best_init_loss:
                    best_init_loss = mean_epoch_loss
                    best_nf_state = self._export_backbones_cpu()
                    best_shift_state = copy.deepcopy(self.shift_syn.detach().to("cpu"))
                    bad_rounds = 0
                else:
                    bad_rounds += 1

                if (ep + 1) % max(1, self.epochs_init // 10) == 0 or ep == 0 or ep == self.epochs_init - 1:
                    save_and_print(
                        self.log_path,
                        f"[Init] epoch {ep + 1}/{self.epochs_init}, recon loss = {mean_epoch_loss:.8f}, "
                        f"best = {best_init_loss:.8f}"
                    )

                if self.init_patience > 0 and bad_rounds >= self.init_patience:
                    save_and_print(
                        self.log_path,
                        f"[Init] Early stop at epoch {ep + 1} "
                        f"(best = {best_init_loss:.8f}, patience = {self.init_patience})"
                    )
                    break

            save_and_print(self.log_path, f"Average recon loss across epochs: {np.mean(epoch_loss_list):.8f}")

            if best_nf_state is not None:
                for c in range(self.num_classes):
                    self.nf_syn[c].load_state_dict(best_nf_state[c])
                self.shift_syn.data.copy_(best_shift_state.to(self.device))
                save_and_print(self.log_path, f"[Init] Restored best initialization, loss = {best_init_loss:.8f}")

        self.label_syn = torch.tensor(
            [np.ones(self.num_per_class) * i for i in range(self.num_classes)],
            requires_grad=False,
            device=self.device,
        ).view(-1).long()

        self.optimizer = self._build_optimizer(for_init=False)
        self.optim_zero_grad()

        os.makedirs(os.path.dirname(initialized_synset_path), exist_ok=True)

        init_save_data = {
            "shared_nf_per_class": self._export_backbones_cpu(),
            "shift": copy.deepcopy(self.shift_syn.detach().to("cpu")),
            "label": copy.deepcopy(self.label_syn.detach().to("cpu")),
        }
        torch.save(init_save_data, initialized_synset_path)
        save_and_print(self.log_path, f"Saved initialized synset at {initialized_synset_path}")

        self.save(name=os.path.basename(initialized_synset_path))

        try:
            self.save_visualization_per_class(
                save_path=f"{self.args.save_path}/imgs/init_per_class.png",
                instance_id=0,
                ncols=2,
                threshold=getattr(self.args, "vis_threshold", 0.6),
            )
            self.save_visualization_each_class(
                save_dir=f"{self.args.save_path}/imgs/init_each_class",
                instance_id=0,
                threshold=getattr(self.args, "vis_threshold", 0.6),
            )
        except Exception as e:
            save_and_print(self.log_path, f"[Visualization Warning] Failed to save init voxel plot: {e}")

        self.show_budget()

    def get(self, indices=None, need_copy=False, detach_backbone=False, grouped=False):
        if indices is None:
            indices = range(len(self.label_syn))
        if not hasattr(indices, "__iter__"):
            indices = [indices]

        indices_list = [int(i) for i in indices]

        volumes_syn = []
        for idx in indices_list:
            class_id = idx // self.num_per_class
            instance_id = idx % self.num_per_class

            shift = self.shift_syn[class_id, instance_id]
            feat_logits = self._forward_instance(class_id, self.coord, shift=shift, detach_backbone=detach_backbone)

            volume_syn = torch.sigmoid(feat_logits).reshape(
                self.depth, self.height, self.width, self.channel
            ).permute(3, 0, 1, 2).contiguous()

            if need_copy:
                volume_syn = volume_syn.detach().clone()

            volumes_syn.append(volume_syn)

        volumes_syn = torch.stack(volumes_syn, dim=0)
        labels_syn = self.label_syn[indices_list]

        if need_copy:
            labels_syn = labels_syn.detach().clone()

        if grouped and len(indices_list) == self.num_classes * self.num_per_class:
            volumes_syn = volumes_syn.view(
                self.num_classes, self.num_per_class, self.channel, self.depth, self.height, self.width
            )
            labels_syn = labels_syn.view(self.num_classes, self.num_per_class)

        return volumes_syn, labels_syn

    def _get_one_index_per_class(self, instance_id=0):
        instance_id = int(instance_id)
        if instance_id >= self.num_per_class:
            instance_id = self.num_per_class - 1
        return [c * self.num_per_class + instance_id for c in range(self.num_classes)]

    def save_visualization(self, save_path, indices=None, max_items=10, ncols=2, threshold=0.5):
        if indices is None:
            if max_items >= self.num_classes:
                indices = self._get_one_index_per_class(instance_id=0)
            else:
                total = len(self.label_syn)
                indices = list(range(min(total, max_items)))
        elif not hasattr(indices, "__iter__"):
            indices = [indices]

        volumes, labels = self.get(indices=indices, need_copy=True, detach_backbone=True)

        save_voxel_grid_3d(
            voxels=volumes,
            save_path=save_path,
            labels=labels,
            ncols=ncols,
            threshold=threshold,
            max_items=min(max_items, len(indices)),
        )
        save_and_print(self.log_path, f"Saved voxel visualization at {save_path}")

    def save_visualization_per_class(self, save_path, instance_id=0, ncols=2, threshold=0.5):
        indices = self._get_one_index_per_class(instance_id=instance_id)
        volumes, labels = self.get(indices=indices, need_copy=True, detach_backbone=True)

        save_voxel_grid_3d(
            voxels=volumes,
            save_path=save_path,
            labels=labels,
            ncols=ncols,
            threshold=threshold,
            max_items=len(indices),
        )
        save_and_print(self.log_path, f"Saved per-class voxel visualization at {save_path}")

    def save_visualization_each_class(self, save_dir, instance_id=0, threshold=0.5):
        os.makedirs(save_dir, exist_ok=True)

        indices = self._get_one_index_per_class(instance_id=instance_id)
        volumes, labels = self.get(indices=indices, need_copy=True, detach_backbone=True)

        for i in range(volumes.shape[0]):
            cls_id = int(labels[i].item())
            out_path = os.path.join(save_dir, f"class_{cls_id:02d}.png")
            save_voxel_single_3d(
                voxel=volumes[i],
                save_path=out_path,
                label=cls_id,
                threshold=threshold,
            )

        save_and_print(self.log_path, f"Saved per-class individual voxel visualizations at {save_dir}")

    def optim_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optim_step(self):
        self.optimizer.step()

    def show_budget(self):
        save_and_print(self.log_path, "=" * 50)
        save_and_print(
            self.log_path,
            f"Allowed Budget Size: {self.num_classes * self.ipc * self.channel * self.depth * self.height * self.width}"
        )

        modulation_budget = self.shift_syn.nelement()
        utilized_budget = (
            sum(sum(t.nelement() for t in tensors) for tensors in (self.nf_syn.parameters(), self.nf_syn.buffers()))
            + modulation_budget
        )

        save_and_print(self.log_path, f"Utilize Budget Size: {utilized_budget}")
        save_and_print(self.log_path, f"Per-class shared backbone params: {self.backbone_params_per_class}")
        save_and_print(self.log_path, f"Total backbone params: {self.backbone_params_total}")
        save_and_print(self.log_path, f"Shift dim per instance: {self.shift_dim_per_instance}")
        save_and_print(self.log_path, f"Latent per instance: {self.latent_params_per_instance}")
        save_and_print(self.log_path, f"Volume size per instance: ({self.channel}, {self.depth}, {self.height}, {self.width})")
        save_and_print(self.log_path, f"Num per class: {self.num_per_class}")
        save_and_print(self.log_path, f"Total num instances: {self.total_num_instances}")

        volumes, _ = self.get(need_copy=True, detach_backbone=True)
        save_and_print(self.log_path, f"Decode condensed data: {volumes.shape}")
        del volumes

        start = time.time()
        self.get(indices=[0])
        single_time = time.time() - start

        start = time.time()
        self.get(indices=[0], need_copy=True, detach_backbone=True)
        single_time_copy = time.time() - start

        save_and_print(self.log_path, f"Single instance retrieval time: {single_time:.5f} {single_time_copy:.5f}")
        save_and_print(self.log_path, "=" * 50)

    def save(self, name, auxiliary=None):
        shared_nf_save = self._export_backbones_cpu()
        shift_save = copy.deepcopy(self.shift_syn.detach().to("cpu"))
        labels_syn_save = copy.deepcopy(self.label_syn.detach().to("cpu"))

        save_data = {
            "shared_nf_per_class": shared_nf_save,
            "shift": shift_save,
            "label": labels_syn_save,
        }
        if isinstance(auxiliary, dict):
            save_data.update(auxiliary)

        torch.save(save_data, f"{self.args.save_path}/{name}")
        save_and_print(self.log_path, f"Saved at {self.args.save_path}/{name}")


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30.0, c=6.0, is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        return self.activation(self.linear(x))


class ModulatedSirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30.0, c=6.0, is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x, shift=None):
        out = self.linear(x)

        if shift is not None:
            if shift.dim() == 1:
                shift = shift.unsqueeze(0)
            out = out + shift

        return self.activation(out)


class SharedModulatedSiren(nn.Module):
    """
    Per-class shared backbone + direct shift modulation.
    code = [shift_1, ..., shift_L]
    """
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        use_checkpoint=True,
    ):
        super().__init__()

        final_activation = nn.Identity() if final_activation is None else final_activation

        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

        self.hidden_layers = nn.ModuleList()
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.hidden_layers.append(
                ModulatedSirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        self.last_layer = SirenLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
        )

    def _split_shift(self, shift):
        shift_list = [None] * self.num_layers
        if shift is not None:
            assert shift.numel() == self.num_layers * self.dim_hidden
            for i in range(self.num_layers):
                shift_list[i] = shift[i * self.dim_hidden:(i + 1) * self.dim_hidden]
        return shift_list

    def _forward_hidden_layers_checkpointed(self, x, shift_list):
        for i, layer in enumerate(self.hidden_layers):
            shift_i = shift_list[i]

            if shift_i is None:
                x = checkpoint(lambda inp, _layer=layer: _layer(inp), x, use_reentrant=False)
            else:
                x = checkpoint(lambda inp, sh, _layer=layer: _layer(inp, shift=sh), x, shift_i, use_reentrant=False)

        return x

    def forward(self, x, shift=None):
        shift_list = self._split_shift(shift)

        if self.use_checkpoint and self.training:
            x = self._forward_hidden_layers_checkpointed(x, shift_list)
        else:
            for i, layer in enumerate(self.hidden_layers):
                x = layer(x, shift=shift_list[i])

        return self.last_layer(x)

# Backward compatibility alias
DDiF = SFRD
