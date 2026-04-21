import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt


# =========================
#  Minimal decoder network
# =========================

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
    ):
        super().__init__()

        final_activation = nn.Identity() if final_activation is None else final_activation

        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

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

    def forward(self, x, shift=None):
        shift_list = self._split_shift(shift)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, shift=shift_list[i])
        return self.last_layer(x)


# =========================
#  SFRD decode utilities
# =========================

def build_coords(res=32, device="cpu"):
    zs = torch.linspace(-1.0, 1.0, steps=res, device=device)
    ys = torch.linspace(-1.0, 1.0, steps=res, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=res, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([grid_z, grid_y, grid_x], dim=-1).reshape(-1, 3)
    return coords


def decode_one_per_class(
    ckpt_path,
    num_layers=4,
    layer_size=80,
    dim_in=3,
    dim_out=1,
    w0_initial=30.0,
    w0=40.0,
    res=32,
    instance_id=0,
    device="cuda",
):
    data = torch.load(ckpt_path, map_location="cpu")

    nf_states = data["shared_nf_per_class"]
    shift = data["shift"]       # [num_classes, num_per_class, shift_dim]
    labels = data["label"]      # [num_classes * num_per_class] 其实这里用不到

    num_classes = shift.shape[0]
    num_per_class = shift.shape[1]

    if instance_id >= num_per_class:
        instance_id = num_per_class - 1

    coords = build_coords(res=res, device=device)

    voxels = []
    out_labels = []

    for c in range(num_classes):
        model = SharedModulatedSiren(
            dim_in=dim_in,
            dim_hidden=layer_size,
            dim_out=dim_out,
            num_layers=num_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=w0_initial,
            w0=w0,
        ).to(device)

        model.load_state_dict(nf_states[c])
        model.eval()

        shift_c = shift[c, instance_id].to(device)

        with torch.no_grad():
            logits = model(coords, shift=shift_c)
            prob = torch.sigmoid(logits).reshape(res, res, res, dim_out).permute(3, 0, 1, 2).contiguous()

        voxels.append(prob.cpu())   # [1, D, H, W]
        out_labels.append(c)

    return voxels, out_labels


# =========================
#  Rendering utilities
# =========================

def reorder_occ(occ_dhw, up_axis="h", flip_up=False):
    """
    occ_dhw: [D, H, W] bool
    Return reordered occupancy for matplotlib ax.voxels, where z-axis is vertical.

    up_axis:
      - 'h': vertical = H  (推荐，通常最像正常站立)
      - 'd': vertical = D
      - 'w': vertical = W
    """
    if up_axis == "h":
        # [H, D, W], vertical is H
        occ = np.transpose(occ_dhw, (1, 0, 2))
    elif up_axis == "d":
        # [D, H, W], vertical is D
        occ = occ_dhw
    elif up_axis == "w":
        # [W, D, H], vertical is W
        occ = np.transpose(occ_dhw, (2, 0, 1))
    else:
        raise ValueError(f"Unknown up_axis: {up_axis}")

    if flip_up:
        occ = occ[::-1, :, :]
    return occ


def make_occ_threshold(voxel, threshold):
    # voxel: [1, D, H, W]
    v = voxel[0].numpy()
    occ = v > threshold
    return occ


def make_occ_percentile(voxel, percentile):
    # voxel: [1, D, H, W]
    v = voxel[0].numpy()
    thr = np.percentile(v.reshape(-1), percentile)
    occ = v > thr
    return occ


def render_grid_voxels(
    voxels,
    labels,
    save_path,
    mode="threshold",
    threshold=0.55,
    percentile=92.0,
    up_axis="h",
    flip_up=False,
    ncols=2,
    elev=16,
    azim=35,
    facecolor="#8B0000",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = len(voxels)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))

    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")

        if mode == "threshold":
            occ_dhw = make_occ_threshold(voxels[i], threshold=threshold)
        elif mode == "percentile":
            occ_dhw = make_occ_percentile(voxels[i], percentile=percentile)
        else:
            raise ValueError(mode)

        occ = reorder_occ(occ_dhw, up_axis=up_axis, flip_up=flip_up)

        ax.voxels(
            occ,
            facecolors=facecolor,
            edgecolor="k",
            linewidth=0.05,
        )

        a, b, c = occ.shape
        ax.set_xlim(0, c)
        ax.set_ylim(0, b)
        ax.set_zlim(0, a)
        ax.set_box_aspect((c, b, a))
        ax.view_init(elev=elev, azim=azim)

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.set_title(f"cls={int(labels[i])}", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def render_each_class_voxels(
    voxels,
    labels,
    save_dir,
    mode="threshold",
    threshold=0.55,
    percentile=92.0,
    up_axis="h",
    flip_up=False,
    elev=16,
    azim=35,
    facecolor="#8B0000",
):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(voxels)):
        fig = plt.figure(figsize=(5.5, 5.0))
        ax = fig.add_subplot(111, projection="3d")

        if mode == "threshold":
            occ_dhw = make_occ_threshold(voxels[i], threshold=threshold)
        else:
            occ_dhw = make_occ_percentile(voxels[i], percentile=percentile)

        occ = reorder_occ(occ_dhw, up_axis=up_axis, flip_up=flip_up)

        ax.voxels(
            occ,
            facecolors=facecolor,
            edgecolor="k",
            linewidth=0.05,
        )

        a, b, c = occ.shape
        ax.set_xlim(0, c)
        ax.set_ylim(0, b)
        ax.set_zlim(0, a)
        ax.set_box_aspect((c, b, a))
        ax.view_init(elev=elev, azim=azim)

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.set_title(f"cls={int(labels[i])}", fontsize=10)

        out_path = os.path.join(save_dir, f"class_{int(labels[i]):02d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=280, bbox_inches="tight")
        plt.close(fig)


# =========================
#  Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--res", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--layer_size", type=int, default=80)
    parser.add_argument("--dim_in", type=int, default=3)
    parser.add_argument("--dim_out", type=int, default=1)
    parser.add_argument("--w0_initial", type=float, default=30.0)
    parser.add_argument("--w0", type=float, default=40.0)

    parser.add_argument("--instance_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--mode", type=str, choices=["threshold", "percentile"], default="threshold")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--percentile", type=float, default=92.0)

    parser.add_argument("--up_axis", type=str, choices=["h", "d", "w"], default="h")
    parser.add_argument("--flip_up", action="store_true")

    parser.add_argument("--elev", type=float, default=16)
    parser.add_argument("--azim", type=float, default=35)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    voxels, labels = decode_one_per_class(
        ckpt_path=args.ckpt,
        num_layers=args.num_layers,
        layer_size=args.layer_size,
        dim_in=args.dim_in,
        dim_out=args.dim_out,
        w0_initial=args.w0_initial,
        w0=args.w0,
        res=args.res,
        instance_id=args.instance_id,
        device=args.device,
    )

    # stats
    with open(os.path.join(args.save_dir, "stats.txt"), "w", encoding="utf-8") as f:
        for i, v in enumerate(voxels):
            arr = v.numpy().reshape(-1)
            f.write(
                f"class={i}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, std={arr.std():.6f}, "
                f"q90={np.percentile(arr, 90):.6f}, q92={np.percentile(arr, 92):.6f}, "
                f"q94={np.percentile(arr, 94):.6f}, q96={np.percentile(arr, 96):.6f}\n"
            )

    render_grid_voxels(
        voxels=voxels,
        labels=labels,
        save_path=os.path.join(args.save_dir, "grid.png"),
        mode=args.mode,
        threshold=args.threshold,
        percentile=args.percentile,
        up_axis=args.up_axis,
        flip_up=args.flip_up,
        ncols=2,
        elev=args.elev,
        azim=args.azim,
    )

    render_each_class_voxels(
        voxels=voxels,
        labels=labels,
        save_dir=os.path.join(args.save_dir, "each_class"),
        mode=args.mode,
        threshold=args.threshold,
        percentile=args.percentile,
        up_axis=args.up_axis,
        flip_up=args.flip_up,
        elev=args.elev,
        azim=args.azim,
    )

    print(f"Done. Saved SFRD visualization to: {args.save_dir}")


if __name__ == "__main__":
    main()