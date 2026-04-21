import numpy as np
import torch
import copy
import time

from utils import save_and_print
from torch.utils.checkpoint import checkpoint
import os
from tqdm import tqdm
from torch import nn
from math import sqrt
from torchvision.utils import save_image


class SFRD():
    def __init__(self, args):
        ### Basic ###
        self.args = args
        self.log_path = args.log_path
        self.channel = args.channel
        self.num_classes = args.num_classes
        self.im_size = args.im_size
        self.device = args.device
        self.ipc = args.ipc

        ### SIREN base config ###
        self.dim_in = args.dim_in
        self.dim_out = args.dim_out
        self.w0_initial = args.w0_initial
        self.w0 = args.w0

        # keep original values as backup/reference
        self.base_num_layers = args.num_layers
        self.base_layer_size = args.layer_size

        # original global learning rates (kept for compatibility)
        self.lr_nf = args.lr_nf
        self.lr_nf_init = args.lr_nf_init

        self.dist = torch.cuda.device_count() > 1

        ### Shared backbone + shift modulation config ###
        self.shared_mode = getattr(args, "shared_mode", "global")
        self.shared_num_layers = getattr(args, "shared_num_layers", self.base_num_layers)
        self.shared_layer_size = getattr(args, "shared_layer_size", self.base_layer_size)

        # use shared config as actual working config
        self.num_layers = self.shared_num_layers
        self.layer_size = self.shared_layer_size

        # modulation config: shift only
        self.modulation_type = getattr(args, "modulation_type", "shift")
        self.shift_init = getattr(args, "shift_init", 0.0)
        self.latent_std = getattr(args, "latent_std", 0.01)

        # training learning rates
        self.lr_nf_backbone = getattr(args, "lr_nf_backbone", self.lr_nf)
        self.lr_nf_shift = getattr(args, "lr_nf_shift", self.lr_nf * 5)

        # initialization-stage learning rates
        self.lr_nf_init_backbone = getattr(args, "lr_nf_init_backbone", self.lr_nf_init)
        self.lr_nf_init_shift = getattr(args, "lr_nf_init_shift", self.lr_nf_init * 5)

        # initialization control
        self.epochs_init = getattr(self.args, "epochs_init", self.args.epochs_init)
        self.init_batch_per_step = getattr(self.args, "init_batch_per_step", 32)

        # how many instances participate in each initialization epoch
        # if <= 0, use all instances in every epoch
        self.init_instances_per_epoch = getattr(self.args, "init_instances_per_epoch", -1)

        # alternating / freezing control
        self.train_backbone = getattr(args, "train_backbone", False)
        self.train_latent = getattr(args, "train_latent", True)

        if (not self.train_backbone) and (not self.train_latent):
            raise ValueError("At least one of train_backbone or train_latent must be True.")

        ### Direct shift code size ###
        self.shift_dim_per_instance = self.num_layers * self.layer_size
        self.latent_dim = self.shift_dim_per_instance

        shared_field_temp = SharedFieldSiren(
            dim_in=self.dim_in,
            dim_hidden=self.layer_size,
            dim_out=self.dim_out,
            num_layers=self.num_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=self.w0_initial,
            w0=self.w0,
        )

        self.backbone_params_global = sum(
            sum(t.nelement() for t in tensors)
            for tensors in (shared_field_temp.parameters(), shared_field_temp.buffers())
        )
        self.latent_params_per_instance = self.latent_dim

        allowed_total_budget = self.num_classes * self.ipc * self.channel * self.im_size[0] * self.im_size[1]

        if self.args.dipc > 0:
            self.num_per_class = self.args.dipc
        else:
            total_num_instances = int(
                (allowed_total_budget - self.backbone_params_global) / self.latent_params_per_instance
            )
            self.num_per_class = total_num_instances // self.num_classes

        self.total_num_instances = self.num_classes * self.num_per_class
        self.total_budget = self.backbone_params_global + self.total_num_instances * self.latent_params_per_instance

        self.budget_per_instance = self.latent_params_per_instance

        if (self.total_budget > allowed_total_budget) or (self.num_per_class < 1):
            msg = (
                f"Invalid Budget: total_budget={self.total_budget}, "
                f"allowed_total_budget={allowed_total_budget}, "
                f"num_per_class={self.num_per_class}, "
                f"backbone_params_global={self.backbone_params_global}, "
                f"latent_params_per_instance={self.latent_params_per_instance}"
            )
            save_and_print(self.log_path, msg)
            raise ValueError(msg)

        del shared_field_temp

        self.shared_field = None
        self.latent_codes = None
        self.synthetic_labels = None
        self.coordinates = None
        self._sync_legacy_aliases()

    def _sync_legacy_aliases(self):
        """
        Temporary backward-compatibility aliases.
        Remove these after all training scripts are migrated to SFRD naming.
        """
        self.nf_syn = self.shared_field
        self.shift_syn = self.latent_codes
        self.label_syn = self.synthetic_labels
        self.coord = self.coordinates

    def _set_requires_grad(self, module_or_param, flag: bool):
        if module_or_param is None:
            return
        if isinstance(module_or_param, torch.nn.Parameter):
            module_or_param.requires_grad_(flag)
        else:
            for p in module_or_param.parameters():
                p.requires_grad_(flag)

    def _apply_train_mode(self):
        """
        According to self.train_backbone / self.train_latent:
        - freeze/unfreeze shared backbone
        - freeze/unfreeze shift latent codes
        """
        self._set_requires_grad(self.shared_field, self.train_backbone)

        if hasattr(self.shared_field, "use_checkpoint"):
            self.shared_field.use_checkpoint = bool(self.train_backbone)

        self.latent_codes.requires_grad_(self.train_latent)

    def _build_optimizer(self, for_init: bool = False):
        param_groups = []

        if for_init:
            lr_backbone = self.lr_nf_init_backbone
            lr_shift = self.lr_nf_init_shift
        else:
            lr_backbone = self.lr_nf_backbone
            lr_shift = self.lr_nf_shift

        if self.train_backbone:
            backbone_params = [p for p in self.shared_field.parameters() if p.requires_grad]
            if len(backbone_params) > 0:
                param_groups.append({"params": backbone_params, "lr": lr_backbone})

        if self.train_latent and self.latent_codes.requires_grad:
            param_groups.append({"params": [self.latent_codes], "lr": lr_shift})

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

    def init(self, images_real, labels_real, indices_class):
        save_and_print(self.log_path, "=" * 50 + "\n SFRD SynSet Initialization")

        criterion = torch.nn.MSELoss().to(self.device)

        ### Initialize Coordinate ###
        image_temp = torch.rand((self.channel, self.im_size[0], self.im_size[1]), device=self.device)
        self.coordinates, _ = to_coordinates_and_features(image_temp)
        self.coordinates = self.coordinates.to(self.device)
        self._sync_legacy_aliases()
        del image_temp

        # ===== Shared backbone: one global backbone =====
        self.shared_field = SharedFieldSiren(
            dim_in=self.dim_in,
            dim_hidden=self.layer_size,
            dim_out=self.dim_out,
            num_layers=self.num_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=self.w0_initial,
            w0=self.w0,
        ).to(self.device)
        self._sync_legacy_aliases()

        # ===== Per-instance shift latent =====
        self.latent_codes = nn.Parameter(
            torch.randn(
                self.num_classes,
                self.num_per_class,
                self.shift_dim_per_instance,
                device=self.device
            ) * self.latent_std + self.shift_init
        )
        self._sync_legacy_aliases()

        # ===== Apply current train mode =====
        self._apply_train_mode()

        dipc = getattr(self.args, "dipc", 0)
        initialized_synset_path = (
            f"../initialized_synset/"
            f"{self.args.dataset}_{self.args.subset}_{self.args.res}_{self.args.model}_{self.args.ipc}ipc_{dipc}dipc/"
            f"init_sharedshift#({self.dim_in},{self.num_layers},{self.layer_size},{self.dim_out})"
            f"_shift{self.shift_dim_per_instance}"
            f"_({self.w0_initial},{self.w0})"
            f"_({self.epochs_init},bkb{self.lr_nf_init_backbone:.0e},sh{self.lr_nf_init_shift:.0e})"
            f"_inst{self.init_instances_per_epoch}_bs{self.init_batch_per_step}.pt"
        )

        if hasattr(self.args, "zca"):
            if self.args.zca:
                initialized_synset_path = f"{initialized_synset_path[:-3]}_ZCA.pt"

        if os.path.isfile(initialized_synset_path):
            save_and_print(self.log_path, f"\n Load from >>>>> {initialized_synset_path} \n")

            data = torch.load(initialized_synset_path)

            if ("shared_nf" in data and "shift" in data) or ("shared_field" in data and "latent_codes" in data):
                shared_nf_state_dict = data["shared_nf"] if "shared_nf" in data else data["shared_field"]
                shift_state = data["shift"] if "shift" in data else data["latent_codes"]

                self.shared_field.load_state_dict(shared_nf_state_dict)

                assert shift_state.shape == (
                    self.num_classes,
                    self.num_per_class,
                    self.shift_dim_per_instance
                )
                self.latent_codes.data.copy_(shift_state.to(self.device))
                self._sync_legacy_aliases()
            else:
                raise ValueError(f"Loaded file is old-format synset and not compatible: {initialized_synset_path}")

        else:
            save_and_print(self.log_path, f"\n No initialized synset >>>>> {initialized_synset_path} \n")

            images_init = []
            class_ids = []
            instance_ids = []

            for c in range(self.num_classes):
                cls_indices = np.array(indices_class[c])
                if len(cls_indices) < self.num_per_class:
                    raise ValueError(
                        f"Class {c} has only {len(cls_indices)} real samples, "
                        f"but num_per_class={self.num_per_class}. "
                        f"Without-replacement initialization is impossible."
                    )

                sampled_indices = np.random.permutation(cls_indices)[:self.num_per_class]
                sampled_images = images_real[sampled_indices]

                images_init.append(sampled_images)
                class_ids.extend([c] * self.num_per_class)
                instance_ids.extend(list(range(self.num_per_class)))

            images_init = torch.cat(images_init, dim=0)
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
                        image = images_init[flat_idx].to(self.device)
                        image_coord, image_value = to_coordinates_and_features(image)

                        c = class_ids[flat_idx].item()
                        k = instance_ids[flat_idx].item()

                        shift = self.latent_codes[c, k]
                        predicted = self.shared_field(image_coord, shift=shift)
                        loss = criterion(predicted, image_value)
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

            vis_images = images_init.detach().clone()

            for ch in range(self.channel):
                vis_images[:, ch] = vis_images[:, ch] * self.args.std[ch] + self.args.mean[ch]

            vis_images = vis_images.clamp(0.0, 1.0)
            save_image(
                vis_images,
                f"{self.args.save_path}/imgs/Selected_for_initialization.png",
                nrow=self.num_per_class
            )
            del vis_images, images_init, class_ids, instance_ids

        ### Initialize Label ###
        self.synthetic_labels = torch.tensor(
            [np.ones(self.num_per_class) * i for i in range(self.num_classes)],
            requires_grad=False,
            device=self.device
        ).view(-1)
        self.synthetic_labels = self.synthetic_labels.long()
        self._sync_legacy_aliases()

        ### Initialize Optimizer ###
        self.optimizer = self._build_optimizer(for_init=False)
        self.optim_zero_grad()

        os.makedirs(os.path.dirname(initialized_synset_path), exist_ok=True)

        init_save_data = {
            "shared_nf": {k: copy.deepcopy(v.to("cpu")) for k, v in self.shared_field.state_dict().items()},
            "shift": copy.deepcopy(self.latent_codes.detach().to("cpu")),
            "label": copy.deepcopy(self.synthetic_labels.detach().to("cpu")),
            "shared_field": {k: copy.deepcopy(v.to("cpu")) for k, v in self.shared_field.state_dict().items()},
            "latent_codes": copy.deepcopy(self.latent_codes.detach().to("cpu")),
            "labels": copy.deepcopy(self.synthetic_labels.detach().to("cpu")),
        }
        torch.save(init_save_data, initialized_synset_path)
        save_and_print(self.log_path, f"Saved initialized synset at {initialized_synset_path}")

        self.save(name=initialized_synset_path.split("/")[-1])
        self.show_budget()

    def get(self, indices=None, need_copy=False, detach_backbone=False):
        if indices is None:
            indices = range(len(self.synthetic_labels))
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        images_syn = []
        for idx in indices:
            idx = int(idx)
            class_id = idx // self.num_per_class
            instance_id = idx % self.num_per_class

            _shift = self.latent_codes[class_id, instance_id]

            if detach_backbone:
                with torch.no_grad():
                    feat = self.shared_field(self.coordinates, shift=_shift)
                _image_syn = feat.reshape(self.im_size[0], self.im_size[1], self.channel).permute(2, 0, 1)
            else:
                feat = self.shared_field(self.coordinates, shift=_shift)
                _image_syn = feat.reshape(self.im_size[0], self.im_size[1], self.channel).permute(2, 0, 1)

            if need_copy:
                _image_syn = _image_syn.detach().clone()

            images_syn.append(_image_syn)

        images_syn = torch.stack(images_syn)
        labels_syn = self.synthetic_labels[indices]

        if need_copy:
            labels_syn = labels_syn.detach().clone()

        return images_syn, labels_syn

    def optim_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optim_step(self):
        self.optimizer.step()

    def show_budget(self):
        save_and_print(self.log_path, '=' * 50)
        save_and_print(
            self.log_path,
            f"Allowed Budget Size: {self.num_classes * self.ipc * self.channel * self.im_size[0] * self.im_size[1]}"
        )

        modulation_budget = self.latent_codes.nelement()

        utilized_budget = (
            sum(sum(t.nelement() for t in tensors) for tensors in (self.shared_field.parameters(), self.shared_field.buffers()))
            + modulation_budget
        )
        save_and_print(self.log_path, f"Utilize Budget Size: {utilized_budget}")
        save_and_print(self.log_path, f"Global shared backbone: {self.backbone_params_global}")
        save_and_print(self.log_path, f"Shift dim per instance: {self.shift_dim_per_instance}")
        save_and_print(self.log_path, f"Latent per instance: {self.latent_params_per_instance}")
        save_and_print(self.log_path, f"Num per class: {self.num_per_class}")
        save_and_print(self.log_path, f"Total num instances: {self.total_num_instances}")

        images, _ = self.get(need_copy=True, detach_backbone=True)
        save_and_print(self.log_path, f"Decode condensed data: {images.shape}")
        del images

        start = time.time()
        self.get(indices=[0])
        single_time = time.time() - start

        start = time.time()
        self.get(indices=[0], need_copy=True, detach_backbone=True)
        single_time_copy = time.time() - start

        save_and_print(self.log_path, f"Single instance retrieval time: {single_time:.5f} {single_time_copy:.5f}")
        save_and_print(self.log_path, '=' * 50)

    def save(self, name, auxiliary=None):
        shared_nf_save = {k: copy.deepcopy(v.to("cpu")) for k, v in self.shared_field.state_dict().items()}
        shift_save = copy.deepcopy(self.latent_codes.detach().to("cpu"))
        labels_syn_save = copy.deepcopy(self.synthetic_labels.detach().to("cpu"))

        save_data = {"shared_nf": shared_nf_save, "shift": shift_save, "label": labels_syn_save, "shared_field": shared_nf_save, "latent_codes": shift_save, "labels": labels_syn_save}
        if type(auxiliary) == dict:
            save_data.update(auxiliary)

        torch.save(save_data, f"{self.args.save_path}/{name}")
        save_and_print(self.log_path, f"Saved at {self.args.save_path}/{name}")

        del shared_nf_save, shift_save, labels_syn_save, save_data


def to_coordinates_and_features(img):
    """
    img: [C, H, W]
    return:
        coordinates: [H*W, 2], each dim normalized to [-1, 1]
        features:    [H*W, C]
    """
    _, h, w = img.shape

    ys = torch.linspace(-1.0, 1.0, steps=h, device=img.device)
    xs = torch.linspace(-1.0, 1.0, steps=w, device=img.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coordinates = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)

    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


class Sine(nn.Module):
    def __init__(self, w0=1.):
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


class Siren(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = SirenLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation
        )

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)


class TranslationTranslationModulatedSirenLayer(nn.Module):
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


class SharedFieldSiren(nn.Module):
    """
    Shared backbone + direct shift modulation.
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
                TranslationModulatedSirenLayer(
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

# ----------------------------------------------------------------------
# Temporary compatibility aliases during repository migration.
# ----------------------------------------------------------------------
DDiF = SFRD
ModulatedSirenLayer = TranslationModulatedSirenLayer
SharedModulatedSiren = SharedFieldSiren
