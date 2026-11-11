
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """LayerNorm over channel dim for NCHW."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

def _dynamic_depthwise_conv2d(x: torch.Tensor, w: torch.Tensor, padding: int):
    """Apply per-sample, per-channel depthwise filters."""
    B, C, H, W = x.shape
    k = w.shape[-1]

    x2 = x.reshape(1, B*C, H, W)
    w2 = w.reshape(B*C, 1, k, k)

    y2 = F.conv2d(x2, w2, bias=None, stride=1, padding=padding, groups=B*C)
    y  = y2.reshape(B, C, H, W)
    return y

def _pad2d(x: torch.Tensor, pad: int, padding_mode: str, constant_value: float) -> torch.Tensor:
    if pad <= 0:
        return x
    pad_args = (pad, pad, pad, pad)
    if padding_mode == "constant":
        return F.pad(x, pad_args, mode="constant", value=constant_value)
    if padding_mode in ("reflect", "replicate", "circular"):
        return F.pad(x, pad_args, mode=padding_mode)
    raise ValueError(
        f"Unsupported padding_mode={padding_mode}. "
        f"Use one of: constant/reflect/replicate/circular."
    )

class ConvFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, bias=True)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfAttention2D(nn.Module):
    """MHSA over spatial tokens."""
    def __init__(self, dim: int, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        x_ = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x_).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out

class SharedLKFull(nn.Module):
    """
    Full large-kernel convolution with channel mixing.

    Padding is handled by F.pad to support constant, reflect, replicate, and
    circular modes.
    """
    def __init__(self, channels: int, k: int = 11, bias: bool = True,
                 padding_mode: str = "constant", constant_value: float = 0.0):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("k must be odd")
        self.pad = k // 2
        self.padding_mode = padding_mode
        self.constant_value = float(constant_value)

        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=k,
            padding=0,
            groups=1,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad2d(x, self.pad, self.padding_mode, self.constant_value)
        return self.conv(x)

class ConvAttn(nn.Module):
    """Partial-channel LK + Dynamic-kernel depthwise."""
    def __init__(
        self,
        dim: int,
        partial_channels: int = 16,
        dk_hidden: int = 32,
    ):
        super().__init__()
        self.p = min(partial_channels, dim)
        self.dk_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.p, dk_hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(dk_hidden, self.p * 9, 1, bias=True),
        )

        self.proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, lk_op: Optional[nn.Module]) -> torch.Tensor:
        x_att = x[:, :self.p]
        x_idt = x[:, self.p:]
        if lk_op is None:
            raise ValueError("SPCNet requires the shared large-kernel operator.")
        w = self.dk_mlp(x_att)
        w = w.view(x.size(0), self.p, 9)
        w = w.softmax(dim=-1).view(x.size(0), self.p, 3, 3)
        x_res = self.act(lk_op(x_att) + _dynamic_depthwise_conv2d(x_att, w, padding=1))
        x_cat = torch.cat([x_res, x_idt], dim=1) if x_idt.numel() > 0 else x_res
        return x + self.proj(x_cat)

class SPCB(nn.Module):
    """Shared-Pattern Collaborative Block (SPCB)."""

    def __init__(
        self,
        dim,
        lk,
        num_heads=4,
        m=2,
        partial_channels=16,
        dk_hidden=32,
        use_self_attn=True,
        ffn_ratio=4,
        drop=0.0,
    ):
        super().__init__()
        self.m, self.lk, self.use_self_attn = m, lk, use_self_attn
        self.norm_in = LayerNorm2d(dim)
        self.pre_ffn = ConvFFN(dim, hidden_dim=dim * ffn_ratio, drop=drop)
        if use_self_attn:
            self.norm_sa = LayerNorm2d(dim)
            self.sa = SelfAttention2D(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.ffns = nn.ModuleList([ConvFFN(dim, dim * ffn_ratio, drop) for _ in range(m)])
        self.attns = nn.ModuleList([
            ConvAttn(
                dim,
                partial_channels,
                dk_hidden,
            )
            for _ in range(m)
        ])
        self.norm_out = LayerNorm2d(dim)
        self.out_conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        f_in = x + self.pre_ffn(self.norm_in(x))

        if self.use_self_attn:
            f_curr = f_in + self.sa(self.norm_sa(f_in))
        else:
            f_curr = f_in

        for ffn, attn in zip(self.ffns, self.attns):
            f_curr = f_curr + ffn(f_curr)
            f_curr = attn(f_curr, self.lk)

        return shortcut + self.out_conv(self.norm_out(f_curr))

class ChannelLayerNorm1d(nn.Module):
    """LayerNorm over channel dim for (B,C,N) tensors."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2)
        return x

class SpatialFilter(nn.Module):
    """Spatial attention filter for refining local feature responses."""
    def __init__(self, channels: int, kernel_size: int = 7, gate_gain: float = 0.25):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.gate_gain = float(gate_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg_out, max_out], dim=1)
        gate = 1.0 + self.gate_gain * torch.tanh(self.conv(s))
        return x * gate

class ChangeMemoryModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.norm = ChannelLayerNorm1d(2 * channels)
        self.embed = nn.Conv1d(2 * channels, 2 * channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.mix = nn.Conv1d(2 * channels, 2, kernel_size=1, bias=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = f1.shape

        cross_f = torch.cat([f1, f2], dim=1)

        dre = cross_f.view(b, 2 * c, h * w)
        dembed = self.act(self.embed(self.norm(dre)))
        d1, d2 = dembed.chunk(2, dim=1)

        if h >= 2 and w >= 2:
            d2_map = d2.view(b, c, h, w)
            d2_p = self.pool(d2_map)
            d2_up = F.interpolate(d2_p, size=(h, w), mode="bilinear", align_corners=False)
            d2_up = d2_up.view(b, c, h * w)
        else:
            d2_up = d2

        mix_in = torch.cat([d1, d2_up], dim=1)

        weight = F.softmax(self.act(self.mix(mix_in)), dim=1)
        weight = weight.view(b, 2, h, w)

        w_sal = weight[:, 0:1, :, :]
        w_nonsal = weight[:, 1:2, :, :]
        return w_sal, w_nonsal

class RAMA(nn.Module):
    """Region-Adaptive Modality Allocator for spatially adaptive fusion."""
    def __init__(
        self,
        channels: int,
        spatial_gate_gain: float = 0.25,
    ):
        super().__init__()
        self.cmm = ChangeMemoryModule(channels)
        hidden = max(channels // 2, 16)

        self.modality_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * channels, kernel_size=1, bias=True),
        )

        self.pre = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, bias=True)
        self.res_proj = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        self.agree_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.diff_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.merge = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
        )

        self.sf = SpatialFilter(channels, gate_gain=spatial_gate_gain)

        self.ab_param = nn.Parameter(torch.zeros(2))
        self.out_act = nn.GELU()

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        return_weights: bool = False,
    ):
        pair = torch.cat([f1, f2], dim=1)
        gate = torch.tanh(self.modality_gate(pair))
        g1, g2 = gate.chunk(2, dim=1)
        f1 = f1 * (1.0 + g1)
        f2 = f2 * (1.0 + g2)

        w_sal, w_nonsal = self.cmm(f1, f2)

        x = self.pre(torch.cat([f1, f2], dim=1))
        f_sal, f_nonsal = x.chunk(2, dim=1)

        f_esal = self.sf(f_sal) * w_sal
        f_enonsal = self.sf(f_nonsal) * w_nonsal

        ab = F.softmax(self.ab_param, dim=0)
        out = ab[0] * f_esal + ab[1] * f_enonsal
        out = out + self.res_proj(torch.cat([f1, f2], dim=1))
        f_agree = self.agree_proj(f1 + f2)
        f_diff = self.diff_proj(torch.abs(f1 - f2))
        out = out + self.merge(torch.cat([out, f_agree, f_diff], dim=1))
        out = self.out_act(out)
        if return_weights:
            return out, w_sal, w_nonsal
        return out

class FeatureRefiner(nn.Module):
    """Lightweight modality-specific refinement before cross-modal fusion."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = max(dim // 4, 16)
        self.norm = LayerNorm2d(dim)
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(x)
        x = x * self.gate(residual)
        return residual + x

class SPCNet(nn.Module):
    """
    Shared-Pattern Collaborative Network for HSI and auxiliary-modality
    classification. The model accepts PCA-reduced HSI patches and SAR/LiDAR/MSI
    patches through the common ``net(hsi_pca, aux)`` backbone interface.
    """

    _LEGACY_STATE_PREFIXES = {
        "trunk.": "shared_spcb_blocks.",
        "hsi_trunk.": "hsi_spcb_blocks.",
        "sar_trunk.": "aux_spcb_blocks.",
        "fusion.": "rama.",
        "hsi_refine.": "hsi_feature_refiner.",
        "sar_refine.": "aux_feature_refiner.",
        "tail_conv.": "refinement.",
    }

    def __init__(
        self,
        hsi_dim: int,
        sar_dim: int,
        num_classes: int,
        embed_dim: int = 64,
        depth: int = 4,
        block_m: int = 2,
        num_heads: int = 4,
        partial_channels: int = 16,
        dk_hidden: int = 32,
        lk_kernel: int = 11,
        use_self_attn: bool = True,
        dropout: float = 0.0,
        lk_padding_mode: str = "reflect",
        spatial_gate_gain: float = 0.25,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.hsi_stem = nn.Conv2d(hsi_dim, embed_dim, kernel_size=3, padding=1, bias=True)
        self.sar_stem = nn.Conv2d(sar_dim, embed_dim, kernel_size=3, padding=1, bias=True)
        self.lk = SharedLKFull(
            min(partial_channels, embed_dim),
            k=lk_kernel,
            padding_mode=lk_padding_mode,
            constant_value=0.0,
        )

        def make_block():
            return SPCB(
                dim=embed_dim,
                lk=self.lk,
                num_heads=num_heads,
                m=block_m,
                partial_channels=partial_channels,
                dk_hidden=dk_hidden,
                use_self_attn=use_self_attn,
                drop=0.0,
            )

        self.hsi_spcb_blocks = nn.ModuleList(make_block() for _ in range(depth))
        self.aux_spcb_blocks = nn.ModuleList(make_block() for _ in range(depth))
        self.hsi_feature_refiner = FeatureRefiner(embed_dim)
        self.aux_feature_refiner = FeatureRefiner(embed_dim)
        self.rama = RAMA(channels=embed_dim, spatial_gate_gain=spatial_gate_gain)

        self.refinement = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True)
        self.norm = LayerNorm2d(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.hsi_aux_head = nn.Linear(embed_dim, num_classes)
        self.sar_aux_head = nn.Linear(embed_dim, num_classes)
        self.head = nn.Linear(embed_dim, num_classes)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for legacy_name, current_name in self._LEGACY_STATE_PREFIXES.items():
            legacy_prefix = prefix + legacy_name
            current_prefix = prefix + current_name
            for key in tuple(state_dict):
                if key.startswith(legacy_prefix):
                    current_key = current_prefix + key[len(legacy_prefix):]
                    state_dict.setdefault(current_key, state_dict[key])
                    del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def hsi_trunk(self):
        return self.hsi_spcb_blocks

    @property
    def sar_trunk(self):
        return self.aux_spcb_blocks

    @property
    def fusion(self):
        return self.rama

    @property
    def hsi_refine(self):
        return self.hsi_feature_refiner

    @property
    def sar_refine(self):
        return self.aux_feature_refiner

    @property
    def tail_conv(self):
        return self.refinement

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:

        if x is None:
            raise ValueError("Input is None")
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() == 4:
            return x
        if x.dim() == 3:

            return x.unsqueeze(1)
        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    def _forward_branch(self, x: torch.Tensor, stem: nn.Module, spcb_blocks: nn.ModuleList) -> torch.Tensor:
        f0 = stem(x)
        f = f0
        for blk in spcb_blocks:
            f = blk(f)
        return f + f0

    def forward_features(self, hsi_pca: torch.Tensor, sar: torch.Tensor):
        """Return branch/fused feature maps without changing training outputs."""
        hsi = self._to_nchw(hsi_pca)
        sar = self._to_nchw(sar)

        fh = self._forward_branch(hsi, self.hsi_stem, self.hsi_spcb_blocks)
        fs = self._forward_branch(sar, self.sar_stem, self.aux_spcb_blocks)

        fh = self.hsi_feature_refiner(fh)
        fs = self.aux_feature_refiner(fs)

        f, ah, ax = self.rama(fh, fs, return_weights=True)
        f = f + self.refinement(f)
        f = self.norm(f)

        return {
            "hsi_feature": fh,
            "aux_feature": fs,
            "fused_feature": f,
            "Ah": ah,
            "Ax": ax,
        }

    def forward(self, hsi_pca: torch.Tensor, sar: torch.Tensor):
        features = self.forward_features(hsi_pca, sar)
        fh = features["hsi_feature"]
        fs = features["aux_feature"]
        f = features["fused_feature"]

        _, _, H, W = f.shape
        f_cls = f[:, :, H // 2, W // 2]
        f_cls = self.drop(f_cls)
        logits = self.head(f_cls)

        if self.training:
            fh_cls = self.drop(fh[:, :, H // 2, W // 2])
            fs_cls = self.drop(fs[:, :, H // 2, W // 2])
            return {
                "logits_main": logits,
                "logits_hsi": self.hsi_aux_head(fh_cls),
                "logits_sar": self.sar_aux_head(fs_cls),
            }

        return logits
