# nf_hashfourier.py
import math
import torch
from torch import nn
import torch.nn.functional as F


# =========================================================
# Basic encoders
# =========================================================

class FourierEncoder(nn.Module):
    """Fixed Fourier features (no learnable params)."""

    def __init__(self, K: int = 1):
        super().__init__()
        self.K = K
        self.register_buffer(
            "freqs",
            (2.0 ** torch.arange(K)) * math.pi,
            persistent=False,
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy: [N, 2] in [-1, 1]
        return: [N, 4K]
        """
        x = xy.unsqueeze(-1) * self.freqs  # [N, 2, K]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).reshape(xy.shape[0], -1)


def fast_hash_2d(ix: torch.Tensor, iy: torch.Tensor, table_size: int) -> torch.Tensor:
    return ((ix * 73856093) ^ (iy * 19349663)) % table_size


class HashGridEncoder2D(nn.Module):
    """Multi-level 2D hash-grid with bilinear interpolation (shared embeddings)."""

    def __init__(
        self,
        levels: int = 2,
        features_per_level: int = 1,
        base_res: int = 8,
        per_level_scale: float = 2.0,
        table_size: int = 1024,
    ):
        super().__init__()
        self.levels = levels
        self.features_per_level = features_per_level
        self.base_res = base_res
        self.per_level_scale = per_level_scale
        self.table_size = table_size

        self.emb = nn.Parameter(torch.empty(levels, table_size, features_per_level))
        nn.init.uniform_(self.emb, -1e-4, 1e-4)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy: [N, 2] in [-1, 1]
        return: [N, levels * F]
        """
        u = (xy + 1.0) * 0.5
        u = torch.clamp(u, 0.0, 1.0)

        feats = []
        for l in range(self.levels):
            res = int(self.base_res * (self.per_level_scale ** l))  # e.g. 8, 16, 32, ...
            p = u * (res - 1)

            x, y = p[:, 0], p[:, 1]
            x0 = torch.floor(x).long()
            y0 = torch.floor(y).long()
            x1 = torch.clamp(x0 + 1, max=res - 1)
            y1 = torch.clamp(y0 + 1, max=res - 1)

            wx = (x - x0.float()).unsqueeze(-1)  # [N, 1]
            wy = (y - y0.float()).unsqueeze(-1)

            h00 = fast_hash_2d(x0, y0, self.table_size)
            h10 = fast_hash_2d(x1, y0, self.table_size)
            h01 = fast_hash_2d(x0, y1, self.table_size)
            h11 = fast_hash_2d(x1, y1, self.table_size)

            e00 = self.emb[l, h00]  # [N, F]
            e10 = self.emb[l, h10]
            e01 = self.emb[l, h01]
            e11 = self.emb[l, h11]

            ex0 = e00 * (1 - wx) + e10 * wx
            ex1 = e01 * (1 - wx) + e11 * wx
            e = ex0 * (1 - wy) + ex1 * wy

            feats.append(e)

        return torch.cat(feats, dim=-1)


class CoordEncoder(nn.Module):
    """Concat Fourier + Hash-grid."""

    def __init__(
        self,
        fourier_K: int = 1,
        hash_levels: int = 2,
        hash_F: int = 1,
        base_res: int = 8,
        scale: float = 2.0,
        table_size: int = 1024,
    ):
        super().__init__()
        self.fourier = FourierEncoder(K=fourier_K)
        self.hashgrid = HashGridEncoder2D(
            levels=hash_levels,
            features_per_level=hash_F,
            base_res=base_res,
            per_level_scale=scale,
            table_size=table_size,
        )

    @property
    def out_dim(self) -> int:
        return 4 * self.fourier.K + self.hashgrid.levels * self.hashgrid.features_per_level

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.fourier(xy), self.hashgrid(xy)], dim=-1)


# =========================================================
# Activations
# =========================================================

def get_act(act: str):
    act = act.lower()
    if act == "silu":
        return nn.SiLU
    if act == "relu":
        return nn.ReLU
    raise ValueError(f"Unsupported activation: {act}")


# =========================================================
# HashFourier branch
# =========================================================

class TinyMLP(nn.Module):
    """Per-instance tiny decoder for HashFourierField."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 6,
        num_hidden_layers: int = 2,
        out_dim: int = 3,
        act: str = "silu",
    ):
        super().__init__()
        assert num_hidden_layers >= 1, "num_hidden_layers must be >= 1"

        act_layer = get_act(act)
        layers = [nn.Linear(in_dim, hidden), act_layer()]

        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), act_layer()]

        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

        self.in_dim = in_dim
        self.hidden = hidden
        self.num_hidden_layers = num_hidden_layers
        self.out_dim = out_dim
        self.act_name = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HashFourierField(nn.Module):
    """Shared encoder + per-instance tiny MLP."""

    def __init__(
        self,
        encoder: CoordEncoder,
        hidden: int = 6,
        num_hidden_layers: int = 2,
        out_dim: int = 3,
        act: str = "silu",
    ):
        super().__init__()
        self.encoder = encoder
        self.mlp = TinyMLP(
            in_dim=encoder.out_dim,
            hidden=hidden,
            num_hidden_layers=num_hidden_layers,
            out_dim=out_dim,
            act=act,
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        z = self.encoder(xy)
        return self.mlp(z)


# =========================================================
# CAM branch
# =========================================================

class CAM2D(nn.Module):
    """
    Shared coordinate-aware modulation (channel-wise).

    Each hidden layer has a gamma/beta grid of shape [hidden, grid_res, grid_res].
    This version supports arbitrary num_layers >= 1.
    """

    def __init__(
        self,
        grid_res: int = 16,
        num_layers: int = 2,
        hidden: int = 6,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.grid_res = grid_res
        self.num_layers = num_layers
        self.hidden = hidden

        # shape: [L, C, H, W]
        self.gamma = nn.Parameter(torch.ones(num_layers, hidden, grid_res, grid_res))
        self.beta = nn.Parameter(torch.zeros(num_layers, hidden, grid_res, grid_res))

    def forward(self, xy: torch.Tensor, layer_idx: int):
        """
        xy: [N, 2] in [-1, 1]
        return:
            gamma, beta: each [N, C]
        """
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx={layer_idx} out of range for num_layers={self.num_layers}")

        grid = xy.view(1, -1, 1, 2)  # [1, N, 1, 2]

        g = F.grid_sample(
            self.gamma[layer_idx:layer_idx + 1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [1, C, N, 1]

        b = F.grid_sample(
            self.beta[layer_idx:layer_idx + 1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [1, C, N, 1]

        g = g.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()  # [N, C]
        b = b.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()  # [N, C]
        return g, b


class TinyMLP_IN2(nn.Module):
    """
    Per-instance tiny decoder with direct xy input.

    Supports arbitrary num_hidden_layers >= 1:
        num_hidden_layers=1: 2 -> hidden -> out
        num_hidden_layers=2: 2 -> hidden -> hidden -> out
        num_hidden_layers=3: 2 -> hidden -> hidden -> hidden -> out
        ...
    """

    def __init__(
        self,
        hidden: int = 6,
        num_hidden_layers: int = 2,
        out_dim: int = 3,
        act: str = "silu",
    ):
        super().__init__()
        assert num_hidden_layers >= 1, "num_hidden_layers must be >= 1"

        act_layer = get_act(act)
        self.act = act_layer()

        self.in_layer = nn.Linear(2, hidden)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num_hidden_layers - 1)]
        )
        self.out_layer = nn.Linear(hidden, out_dim)

        # backward-compatible aliases for old code / debugging
        self.fc1 = self.in_layer
        self.fc2 = self.hidden_layers[0] if len(self.hidden_layers) >= 1 else None
        self.fc3 = self.out_layer

        self.hidden = hidden
        self.num_hidden_layers = num_hidden_layers
        self.out_dim = out_dim
        self.act_name = act

    def forward_features(self, x: torch.Tensor):
        """
        Return list of hidden features after activation.
        """
        hs = []

        h = self.act(self.in_layer(x))
        hs.append(h)

        for layer in self.hidden_layers:
            h = self.act(layer(h))
            hs.append(h)

        return hs

    def forward(self, x: torch.Tensor):
        hs = self.forward_features(x)
        out = self.out_layer(hs[-1])
        return out, hs


class CAMField(nn.Module):
    """
    Per-instance MLP with shared coordinate-aware modulation.

    Important:
    - shared_cam is intentionally NOT registered as a child module
      to avoid counting shared params multiple times across instances.
    """

    def __init__(
        self,
        shared_cam: CAM2D,
        hidden: int = 6,
        num_hidden_layers: int = None,
        out_dim: int = 3,
        act: str = "silu",
    ):
        super().__init__()

        # keep backward compatibility:
        # if caller does not pass num_hidden_layers, inherit from shared_cam
        if num_hidden_layers is None:
            num_hidden_layers = shared_cam.num_layers

        if shared_cam.num_layers != num_hidden_layers:
            raise ValueError(
                f"CAMField num_hidden_layers ({num_hidden_layers}) must match "
                f"shared_cam.num_layers ({shared_cam.num_layers})."
            )

        object.__setattr__(self, "shared_cam", shared_cam)

        self.mlp = TinyMLP_IN2(
            hidden=hidden,
            num_hidden_layers=num_hidden_layers,
            out_dim=out_dim,
            act=act,
        )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_hidden_layers)])

        # backward-compatible aliases for 2-layer old code
        self.ln1 = self.norms[0]
        self.ln2 = self.norms[1] if len(self.norms) > 1 else None

        self.hidden = hidden
        self.num_hidden_layers = num_hidden_layers
        self.out_dim = out_dim
        self.act_name = act

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy: [N, 2] in [-1, 1]
        """
        # first hidden layer
        h = self.mlp.act(self.mlp.in_layer(xy))
        g, b = self.shared_cam(xy, layer_idx=0)
        h = g * self.norms[0](h) + b

        # remaining hidden layers
        for i, layer in enumerate(self.mlp.hidden_layers, start=1):
            h = self.mlp.act(layer(h))
            g, b = self.shared_cam(xy, layer_idx=i)
            h = g * self.norms[i](h) + b

        out = self.mlp.out_layer(h)
        return out
