import numpy as np
import torch
import copy
import time
import os

from utils import save_and_print
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from torch import nn
from math import sqrt
from torchvision.utils import save_image, make_grid


def to_coordinates_and_features(video):
    """
    Accept video in one of:
      [T, C, H, W]
      [C, T, H, W]
      [T, H, W, C]

    Return:
      coordinates: [T*H*W, 3]  for (t, y, x), normalized to [-1, 1]
      features:    [T*H*W, C]
      video_tc_hw: [T, C, H, W]
    """
    if video is None:
        raise ValueError("to_coordinates_and_features got None video")

    if not torch.is_tensor(video):
        video = torch.tensor(video)

    video = video.float()

    if video.dim() != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {tuple(video.shape)}")

    if video.shape[1] in (1, 3):
        video_tc_hw = video.contiguous()
    elif video.shape[0] in (1, 3):
        video_tc_hw = video.permute(1, 0, 2, 3).contiguous()
    elif video.shape[-1] in (1, 3):
        video_tc_hw = video.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(
            f"Cannot infer channel dimension from video shape {tuple(video.shape)}. "
            f"Expected one of [T,C,H,W], [C,T,H,W], [T,H,W,C]."
        )

    t, c, h, w = video_tc_hw.shape

    ts = torch.linspace(-1.0, 1.0, steps=t, device=video_tc_hw.device)
    ys = torch.linspace(-1.0, 1.0, steps=h, device=video_tc_hw.device)
    xs = torch.linspace(-1.0, 1.0, steps=w, device=video_tc_hw.device)

    grid_t, grid_y, grid_x = torch.meshgrid(ts, ys, xs, indexing='ij')
    coordinates = torch.stack([grid_t, grid_y, grid_x], dim=-1).reshape(-1, 3)
    features = video_tc_hw.permute(0, 2, 3, 1).reshape(-1, c).contiguous()

    if coordinates is None or features is None:
        raise RuntimeError("to_coordinates_and_features produced None outputs")

    return coordinates, features, video_tc_hw


def _default_frame_ids(num_frames, show_frames=None):
    # SFRD-style video visualization: show all frames by default.
    if show_frames is None or show_frames >= num_frames:
        return list(range(num_frames))
    show_frames = min(show_frames, num_frames)
    return np.linspace(0, num_frames - 1, num=show_frames, dtype=int).tolist()


def _denorm_videos(videos, mean, std):
    videos = videos.detach().cpu().clone()
    for ch in range(videos.shape[2]):
        videos[:, :, ch] = videos[:, :, ch] * std[ch] + mean[ch]
    return videos.clamp(0.0, 1.0)


def _auto_tile_nrow(num_frames_shown):
    return int(np.ceil(np.sqrt(num_frames_shown)))


def _auto_global_nrow(num_videos):
    # Match the typical DDiF overview layout more closely.
    if num_videos == 50:
        return 10
    if num_videos in [100, 101]:
        return 10
    return int(np.ceil(np.sqrt(num_videos)))


def videos_to_montage_tiles(videos, frame_ids=None, tile_nrow=None, padding=1, pad_value=0.0):
    """
    videos: [N, T, C, H, W]
    return:
      tiles: [N, C, H_tile, W_tile]
      frame_ids: selected frame ids
    """
    assert videos.dim() == 5, f"Expected [N,T,C,H,W], got {videos.shape}"
    n, t, c, h, w = videos.shape

    if frame_ids is None:
        frame_ids = _default_frame_ids(t, show_frames=None)
    frame_ids = [int(fid) for fid in frame_ids]

    if tile_nrow is None:
        tile_nrow = _auto_tile_nrow(len(frame_ids))

    tiles = []
    for i in range(n):
        frames = videos[i, frame_ids]  # [K, C, H, W]
        tile = make_grid(frames, nrow=tile_nrow, padding=padding, pad_value=pad_value)
        tiles.append(tile)

    tiles = torch.stack(tiles, dim=0)
    return tiles, frame_ids


def save_video_grid(
    videos,
    save_path,
    mean,
    std,
    frame_ids=None,
    nrow=None,
    tile_nrow=None,
    outer_padding=2,
    inner_padding=1,
    pad_value=0.0,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    videos = _denorm_videos(videos, mean=mean, std=std)
    tiles, frame_ids = videos_to_montage_tiles(
        videos,
        frame_ids=frame_ids,
        tile_nrow=tile_nrow,
        padding=inner_padding,
        pad_value=pad_value,
    )

    if nrow is None:
        nrow = _auto_global_nrow(tiles.shape[0])

    save_image(tiles, save_path, nrow=nrow, padding=outer_padding, pad_value=pad_value)
    return frame_ids


class SFRD:
    def __init__(self, args):
        self.args = args
        self.log_path = args.log_path
        self.channel = args.channel
        self.num_classes = args.num_classes
        self.im_size = args.im_size
        self.device = args.device
        self.ipc = args.ipc
        self.frames = getattr(args, "frames", 16)

        self.dim_in = args.dim_in
        self.dim_out = args.dim_out
        self.w0_initial = args.w0_initial
        self.w0 = args.w0

        if self.dim_in != 3:
            raise ValueError(
                f"For video distillation, dim_in must be 3 for (t,y,x), but got dim_in={self.dim_in}"
            )

        self.base_num_layers = args.num_layers
        self.base_layer_size = args.layer_size

        self.lr_nf = args.lr_nf
        self.lr_nf_init = args.lr_nf_init

        self.dist = torch.cuda.device_count() > 1

        # one shared backbone per class + per-instance shift modulation
        self.shared_mode = getattr(args, "shared_mode", "per_class")
        self.shared_num_layers = getattr(args, "shared_num_layers", self.base_num_layers)
        self.shared_layer_size = getattr(args, "shared_layer_size", self.base_layer_size)

        self.num_layers = self.shared_num_layers
        self.layer_size = self.shared_layer_size

        self.modulation_type = getattr(args, "modulation_type", "shift")
        if self.modulation_type != "shift":
            raise ValueError(f"Only shift modulation is supported, but got modulation_type={self.modulation_type}")

        self.shift_init = getattr(args, "shift_init", 0.0)
        self.latent_std = getattr(args, "latent_std", 0.01)

        self.lr_nf_backbone = getattr(args, "lr_nf_backbone", self.lr_nf)
        self.lr_nf_shift = getattr(args, "lr_nf_shift", self.lr_nf * 5)

        self.lr_nf_init_backbone = getattr(args, "lr_nf_init_backbone", self.lr_nf_init)
        self.lr_nf_init_shift = getattr(args, "lr_nf_init_shift", self.lr_nf_init * 5)

        self.epochs_init = getattr(self.args, "epochs_init", 5000)
        self.init_batch_per_step = getattr(self.args, "init_batch_per_step", 32)
        self.init_instances_per_epoch = getattr(self.args, "init_instances_per_epoch", -1)

        self.train_backbone = getattr(args, "train_backbone", False)
        self.train_latent = getattr(args, "train_latent", True)

        if (not self.train_backbone) and (not self.train_latent):
            raise ValueError("At least one of train_backbone or train_latent must be True.")

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
            self.num_classes * self.ipc * self.frames * self.channel * self.im_size[0] * self.im_size[1]
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
        self.budget_per_instance = self.latent_params_per_instance

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

    def _set_requires_grad(self, module_or_param, flag: bool):
        if module_or_param is None:
            return
        if isinstance(module_or_param, torch.nn.Parameter):
            module_or_param.requires_grad_(flag)
        else:
            for p in module_or_param.parameters():
                p.requires_grad_(flag)

    def _set_checkpoint_flag(self, flag: bool):
        if not hasattr(self, "nf_syn"):
            return

        if isinstance(self.nf_syn, nn.ModuleList):
            for backbone in self.nf_syn:
                if hasattr(backbone, "use_checkpoint"):
                    backbone.use_checkpoint = flag
        else:
            if hasattr(self.nf_syn, "use_checkpoint"):
                self.nf_syn.use_checkpoint = flag

    def _apply_train_mode(self):
        self._set_requires_grad(self.nf_syn, self.train_backbone)
        self._set_checkpoint_flag(bool(self.train_backbone))
        self.shift_syn.requires_grad_(self.train_latent)

    def _build_optimizer(self, for_init: bool = False):
        param_groups = []

        if for_init:
            lr_backbone = self.lr_nf_init_backbone
            lr_shift = self.lr_nf_init_shift
        else:
            lr_backbone = self.lr_nf_backbone
            lr_shift = self.lr_nf_shift

        if self.train_backbone:
            backbone_params = [p for p in self.nf_syn.parameters() if p.requires_grad]
            if len(backbone_params) > 0:
                param_groups.append({"params": backbone_params, "lr": lr_backbone})

        if self.train_latent and self.shift_syn.requires_grad:
            param_groups.append({"params": [self.shift_syn], "lr": lr_shift})

        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found when building optimizer.")

        return torch.optim.Adam(param_groups)

    def switch_train_stage(self, train_backbone: bool, train_latent: bool, rebuild_optimizer: bool = True):
        self.train_backbone = train_backbone
        self.train_latent = train_latent

        if (not self.train_backbone) and (not self.train_latent):
            raise ValueError("At least one of train_backbone or train_latent must be True.")

        self._apply_train_mode()

        if rebuild_optimizer:
            self.optimizer = self._build_optimizer(for_init=False)
            self.optim_zero_grad()

        save_and_print(
            self.log_path,
            f"[Switch Stage] train_backbone={self.train_backbone}, train_latent={self.train_latent}"
        )

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

    def init(self, videos_real, labels_real, indices_class):
        save_and_print(self.log_path, "=" * 50 + "\n SynSet Initialization")

        criterion = torch.nn.MSELoss().to(self.device)

        video_temp = torch.rand((self.frames, self.channel, self.im_size[0], self.im_size[1]), device=self.device)
        self.coord, _, _ = to_coordinates_and_features(video_temp)
        self.coord = self.coord.to(self.device)
        del video_temp

        self.nf_syn = nn.ModuleList([self._build_single_backbone() for _ in range(self.num_classes)])

        self.shift_syn = nn.Parameter(
            torch.randn(
                self.num_classes,
                self.num_per_class,
                self.shift_dim_per_instance,
                device=self.device
            ) * self.latent_std + self.shift_init
        )

        self._apply_train_mode()

        dipc = getattr(self.args, "dipc", 0)
        subset = getattr(self.args, "subset", "none")
        res = getattr(self.args, "res", f"{self.im_size[0]}x{self.im_size[1]}x{self.frames}")

        initialized_synset_path = (
            f"../initialized_synset/"
            f"{self.args.dataset}_{subset}_{res}_{self.args.model}_{self.args.ipc}ipc_{dipc}dipc/"
            f"init_perclasssharedshift#({self.dim_in},{self.num_layers},{self.layer_size},{self.dim_out})"
            f"_shift{self.shift_dim_per_instance}"
            f"_({self.w0_initial},{self.w0})"
            f"_({self.epochs_init},bkb{self.lr_nf_init_backbone:.0e},sh{self.lr_nf_init_shift:.0e})"
            f"_inst{self.init_instances_per_epoch}_bs{self.init_batch_per_step}.pt"
        )

        if hasattr(self.args, "zca") and self.args.zca:
            initialized_synset_path = f"{initialized_synset_path[:-3]}_ZCA.pt"

        if os.path.isfile(initialized_synset_path):
            save_and_print(self.log_path, f"\n Load from >>>>> {initialized_synset_path} \n")
            data = torch.load(initialized_synset_path, map_location="cpu")

            if "shared_nf_per_class" in data and "shift" in data:
                nf_states = data["shared_nf_per_class"]
                if len(nf_states) != self.num_classes:
                    raise ValueError(
                        f"shared_nf_per_class length mismatch: got {len(nf_states)}, expected {self.num_classes}"
                    )
                for c in range(self.num_classes):
                    self.nf_syn[c].load_state_dict(nf_states[c])

                shift_state = data["shift"]
                assert shift_state.shape == (
                    self.num_classes,
                    self.num_per_class,
                    self.shift_dim_per_instance
                ), f"shift shape mismatch: got {shift_state.shape}"

                self.shift_syn.data.copy_(shift_state.to(self.device))

            elif "shared_nf" in data and "shift" in data:
                save_and_print(
                    self.log_path,
                    "[Load Warning] Found old single-backbone checkpoint. Copying the same backbone to all classes."
                )
                for c in range(self.num_classes):
                    self.nf_syn[c].load_state_dict(data["shared_nf"])

                shift_state = data["shift"]
                assert shift_state.shape == (
                    self.num_classes,
                    self.num_per_class,
                    self.shift_dim_per_instance
                ), f"shift shape mismatch: got {shift_state.shape}"

                self.shift_syn.data.copy_(shift_state.to(self.device))
            else:
                raise ValueError(f"Incompatible synset format: {initialized_synset_path}")

        else:
            save_and_print(self.log_path, f"\n No initialized synset >>>>> {initialized_synset_path} \n")

            videos_init = []
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
                sampled_videos = videos_real[sampled_indices]
                videos_init.append(sampled_videos)
                class_ids.extend([c] * self.num_per_class)
                instance_ids.extend(list(range(self.num_per_class)))

            videos_init = torch.cat(videos_init, dim=0)
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
                        video = videos_init[flat_idx].to(self.device)
                        video_coord, video_value, video = to_coordinates_and_features(video)

                        if video_value is None:
                            raise RuntimeError(
                                f"video_value is None at flat_idx={flat_idx}, "
                                f"original video shape={tuple(videos_init[flat_idx].shape)}"
                            )

                        c = class_ids[flat_idx].item()
                        k = instance_ids[flat_idx].item()

                        shift = self.shift_syn[c, k]
                        predicted = self._forward_instance(c, video_coord, shift=shift, detach_backbone=False)

                        if predicted.shape != video_value.shape:
                            raise RuntimeError(
                                f"Shape mismatch before MSE: predicted={tuple(predicted.shape)}, "
                                f"target={tuple(video_value.shape)}, "
                                f"video_canonical_shape={tuple(video.shape)}"
                            )

                        loss = criterion(predicted, video_value)
                        batch_loss = batch_loss + loss

                    batch_loss = batch_loss / batch_flat_idx.numel()
                    batch_loss.backward()
                    init_optimizer.step()

                    batch_losses.append(batch_loss.item())

                mean_epoch_loss = float(np.mean(batch_losses)) if len(batch_losses) > 0 else 0.0
                epoch_loss_list.append(mean_epoch_loss)

                if (ep + 1) % max(1, self.epochs_init // 10) == 0 or ep == 0 or ep == self.epochs_init - 1:
                    save_and_print(
                        self.log_path,
                        f"[Init] epoch {ep + 1}/{self.epochs_init}, recon loss = {mean_epoch_loss:.8f}"
                    )

            save_and_print(self.log_path, f"Average recon loss across epochs: {np.mean(epoch_loss_list):.8f}")

            vis_path = f"{self.args.save_path}/imgs/Selected_for_initialization.png"
            frame_ids = save_video_grid(
                videos_init.detach(),
                vis_path,
                mean=self.args.mean,
                std=self.args.std,
                frame_ids=None,
                nrow=None,
                tile_nrow=None,
                outer_padding=2,
                inner_padding=1,
                pad_value=0.0,
            )
            save_and_print(self.log_path, f"Saved init visualization at {vis_path}, frame_ids={frame_ids}")

            del videos_init, class_ids, instance_ids

        self.label_syn = torch.tensor(
            [np.ones(self.num_per_class) * i for i in range(self.num_classes)],
            requires_grad=False,
            device=self.device
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

        self.save(name=initialized_synset_path.split("/")[-1])
        self.show_budget()

    def get(self, indices=None, need_copy=False, detach_backbone=False, grouped=False):
        if indices is None:
            indices = range(len(self.label_syn))
        if not hasattr(indices, "__iter__"):
            indices = [indices]

        indices_list = [int(i) for i in indices]

        videos_syn = []
        for idx in indices_list:
            class_id = idx // self.num_per_class
            instance_id = idx % self.num_per_class

            shift = self.shift_syn[class_id, instance_id]
            feat = self._forward_instance(class_id, self.coord, shift=shift, detach_backbone=detach_backbone)

            video_syn = feat.reshape(
                self.frames, self.im_size[0], self.im_size[1], self.channel
            ).permute(0, 3, 1, 2).contiguous()

            if need_copy:
                video_syn = video_syn.detach().clone()

            videos_syn.append(video_syn)

        videos_syn = torch.stack(videos_syn, dim=0)
        labels_syn = self.label_syn[indices_list]

        if need_copy:
            labels_syn = labels_syn.detach().clone()

        if grouped and len(indices_list) == self.num_classes * self.num_per_class:
            videos_syn = videos_syn.view(
                self.num_classes, self.num_per_class, self.frames, self.channel, self.im_size[0], self.im_size[1]
            )
            labels_syn = labels_syn.view(self.num_classes, self.num_per_class)

        return videos_syn, labels_syn

    def save_visualization(self, save_path, frame_ids=None):
        videos, _ = self.get(need_copy=True, detach_backbone=True)
        used_frame_ids = save_video_grid(
            videos,
            save_path,
            mean=self.args.mean,
            std=self.args.std,
            frame_ids=frame_ids,   # None -> show all frames by default
            nrow=None,             # auto layout, e.g. 50 classes -> 10 columns
            tile_nrow=None,        # auto tile layout, e.g. 16 frames -> 4x4
            outer_padding=2,
            inner_padding=1,
            pad_value=0.0,
        )
        return used_frame_ids

    def optim_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optim_step(self):
        self.optimizer.step()

    def show_budget(self):
        save_and_print(self.log_path, "=" * 50)
        save_and_print(
            self.log_path,
            f"Allowed Budget Size: {self.num_classes * self.ipc * self.frames * self.channel * self.im_size[0] * self.im_size[1]}"
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
        save_and_print(self.log_path, f"Frames per instance: {self.frames}")
        save_and_print(self.log_path, f"Num per class: {self.num_per_class}")
        save_and_print(self.log_path, f"Total num instances: {self.total_num_instances}")

        videos, _ = self.get(need_copy=True, detach_backbone=True)
        save_and_print(self.log_path, f"Decode condensed data: {videos.shape}")
        del videos

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

        save_data = {"shared_nf_per_class": shared_nf_save, "shift": shift_save, "label": labels_syn_save}
        if isinstance(auxiliary, dict):
            save_data.update(auxiliary)

        torch.save(save_data, f"{self.args.save_path}/{name}")
        save_and_print(self.log_path, f"Saved at {self.args.save_path}/{name}")

        del shared_nf_save, shift_save, labels_syn_save, save_data


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
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
        out = self.linear(x)
        out = self.activation(out)
        return out


class ModulatedSirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
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

            out = self.activation(out)
            return out

    def forward(self, x, shift=None):
        out = self.linear(x)

        if shift is not None:
            if shift.dim() == 1:
                shift = shift.unsqueeze(0)
            out = out + shift

        out = self.activation(out)
        return out


class SharedModulatedSiren(nn.Module):
    """
    Per-class shared backbone + direct shift modulation.
    For each hidden layer:
        shift_l: [dim_hidden]
    direct code format:
        code = [shift_1, ..., shift_L]
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None,
                 use_checkpoint=True):
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
            activation=final_activation
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
                x = checkpoint(
                    lambda inp, _layer=layer: _layer(inp),
                    x,
                    use_reentrant=False
                )
            else:
                x = checkpoint(
                    lambda inp, sh, _layer=layer: _layer(inp, shift=sh),
                    x, shift_i,
                    use_reentrant=False
                )

        return x

    def forward(self, x, shift=None):
        shift_list = self._split_shift(shift)

        if self.use_checkpoint and self.training:
            x = self._forward_hidden_layers_checkpointed(x, shift_list)
        else:
            for i, layer in enumerate(self.hidden_layers):
                x = layer(x, shift=shift_list[i])

        x = self.last_layer(x)
        return x

# Backward compatibility alias
DDiF = SFRD
