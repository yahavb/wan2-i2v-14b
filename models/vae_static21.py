"""Fullgraph Wan2.1 VAE encode/decode on Neuron (single rank, whole frame).

Port of staging vllm-omni-neuron's NeuronWanDecoder3d / NeuronWanEncoder3d
(autoencoder_kl_wan.py) onto the original Wan2.1 VAE in wan/modules/vae2_1.py, with the
static-cache plumbing from ~/wan2-ti2v-5b/models/vae_static.py:

  * Fixed-shape causal cache. Every CausalConv3d cache slot is a zero tensor [B, C, 2, h, w]
    from the start (zero frames == the causal zero padding the original applies), passed
    into the graph as flat inputs and returned as outputs; update = cat(old, x)[:, :, -2:].
    `first_chunk` replaces the 'Rep' / None sentinels. Two graphs per direction: decoder
    first chunk (T=1) / rest; encoder chunk 0 (1 frame) / rest (4 frames).
  * In-graph kernels (staging's dispatch):
      - nkilib conv3d for 3x3x3 stride-1 convs with C_in, C_out >= 96 and D_out >= 2 (the
        2-frame cache makes the conv padding always (0,0,1,1,1,1); D_out=1 regressed 25%
        in staging, so T=1 sites stay on the compiler);
      - the 3x3 pad-1 upsample Conv2d lifted to D=1 on the same kernel (C_in >= 96, C_out > 32);
      - nkilib attention_cte for the single-head mid-block attention (q pre-scaled by C^-0.5).
  * Rewrites (staging): RMS_norm as x * rsqrt(mean(x^2)) in fp32 (F.normalize's guard cost
    ~67us/call); nearest-exact 2x upsample as repeat_interleave (exact for an integer 2x).

All units: torch.compile(backend="neuron", fullgraph=True, dynamic=False).
VAE_NKI_CONV=0 keeps every conv on the compiler; VAE_NKI_ATTN=0 uses SDPA.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wan.modules import vae2_1 as V

logger = logging.getLogger(__name__)

CACHE_T = 2
VAE_NKI_CONV = os.environ.get("VAE_NKI_CONV", "1") == "1"
VAE_NKI_ATTN = os.environ.get("VAE_NKI_ATTN", "1") == "1"
MIN_CH = 96
COMPILE_KW = dict(backend="neuron", dynamic=False, fullgraph=True)


# ─── rewrites (staging NeuronWanRMSNorm / WanUpsample) ─────────────────────────────────

def _rms_forward(self, x):
    xf = x.float()
    dim = 1 if self.channel_first else -1
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=dim, keepdim=True).clamp_min(1e-12))
    return (y * self.gamma.float()).to(x.dtype) + self.bias


def _upsample_forward(self, x):
    return x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def install_rewrites():
    V.RMS_norm.forward = _rms_forward
    V.Upsample.forward = _upsample_forward


# ─── NKI conv3d (staging conv3d_3x3x3 dispatch) ───────────────────────────────────────

_conv_call = None


def _get_conv():
    global _conv_call
    if _conv_call is None:
        import nki
        try:
            from nkilib.experimental.conv.conv3d import conv3d
            src = "image nkilib"
        except ImportError:
            from kernels.nkilib_vendor.experimental.conv.conv3d import conv3d
            src = "kernels/nkilib_vendor"
        from models.fast_dit import _wrap

        @nki.jit
        def _conv3d_kernel(x, filters, bias):
            """x [B, C_in, D, H, W] (causal time padding already applied), filters
            [K_d, K_h, K_w, C_in, C_out], bias [C_out]; spatial padding 1, stride 1."""
            return conv3d(x, filters, bias, stride=(1, 1, 1), padding=(0, 0, 1, 1, 1, 1),
                          dilation=(1, 1, 1))

        _conv_call = _wrap(_conv3d_kernel, "wan14b_vae::conv3d")
        logger.info(f"[vae_static21] nkilib conv3d from {src}")
    return _conv_call


def _pack(weight):
    """[C_out, C_in, kd, kh, kw] -> [kd, kh, kw, C_in, C_out], permuted on CPU."""
    return weight.detach().cpu().permute(2, 3, 4, 1, 0).contiguous().to(weight.device)


def _conv3d_ok(conv):
    return (isinstance(conv, V.CausalConv3d) and tuple(conv.kernel_size) == (3, 3, 3)
            and tuple(conv.stride) == (1, 1, 1) and tuple(conv.dilation) == (1, 1, 1)
            and conv.groups == 1 and conv.bias is not None
            and conv._padding[0] == 1 and conv._padding[2] == 1 and conv._padding[4] == 2
            and conv.in_channels >= MIN_CH and conv.out_channels >= MIN_CH)


def _conv2d_ok(conv):
    return (isinstance(conv, nn.Conv2d) and tuple(conv.kernel_size) == (3, 3)
            and tuple(conv.stride) == (1, 1) and tuple(conv.padding) == (1, 1)
            and conv.bias is not None and conv.in_channels >= MIN_CH and conv.out_channels > 32)


def prepare_nki(vae_model):
    """Pack filters for every eligible site (before compile, so they are graph constants)."""
    if not VAE_NKI_CONV:
        return 0
    _get_conv()
    n = 0
    for m in vae_model.modules():
        if _conv3d_ok(m):
            m.register_buffer("nki_w", _pack(m.weight), persistent=False)
            n += 1
        elif isinstance(m, V.Resample) and m.mode in ("upsample2d", "upsample3d") and _conv2d_ok(m.resample[1]):
            c = m.resample[1]
            c.register_buffer("nki_w", _pack(c.weight.unsqueeze(2)), persistent=False)  # [1,3,3,Ci,Co]
            n += 1
    logger.info(f"[vae_static21] NKI conv sites: {n}")
    return n


# ─── static-cache module walkers ──────────────────────────────────────────────────────

def _cconv(conv, x, cache, idx):
    """Causal conv with a fixed 2-frame cache; updates cache[idx]."""
    i = idx[0]
    old = cache[i]
    if hasattr(conv, "nki_w") and x.shape[2] >= 2:
        y = _get_conv()(torch.cat([old, x], dim=2).contiguous(), conv.nki_w, conv.bias)
    else:
        y = conv(x, old)
    cache[i] = torch.cat([old, x], dim=2)[:, :, -CACHE_T:].contiguous()
    idx[0] += 1
    return y


def _upconv(conv, x):
    """Resample 3x3 Conv2d on [N, C, H, W]."""
    if hasattr(conv, "nki_w"):
        return _get_conv()(x.unsqueeze(2).contiguous(), conv.nki_w, conv.bias).squeeze(2)
    return conv(x)


def _resblock(blk, x, cache, idx):
    h = blk.shortcut(x)
    for layer in blk.residual:
        x = _cconv(layer, x, cache, idx) if isinstance(layer, V.CausalConv3d) else layer(x)
    return x + h


def _attention(blk, x):
    identity = x
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = blk.norm(x)
    qkv = blk.to_qkv(x).reshape(b * t, c * 3, h * w).permute(0, 2, 1)   # [BT, S, 3C]
    q, k, v = qkv.chunk(3, dim=-1)
    if VAE_NKI_ATTN:
        from models.fast_dit import _get_cte
        y = _get_cte()((q * c ** -0.5).contiguous(), k.contiguous(), v.contiguous())  # [BT, S, C]
        y = y.clone()  # staging: keeps the attention output from being fused (precision)
    else:
        y = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
    x = y.permute(0, 2, 1).reshape(b * t, c, h, w)
    x = blk.proj(x)
    return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4) + identity


def _resample_dec(rs, x, cache, idx, first_chunk):
    b, c, t, h, w = x.shape
    if rs.mode == "upsample3d":
        if first_chunk:
            idx[0] += 1                      # original: 'Rep', time_conv skipped, cache stays zero
        else:
            y = _cconv(rs.time_conv, x, cache, idx).reshape(b, 2, c, t, h, w)
            x = torch.stack((y[:, 0], y[:, 1]), 3).reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, x.shape[3], x.shape[4])
    x = _upconv(rs.resample[1], rs.resample[0](x))
    return x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)


def _resample_enc(rs, x, cache, idx, first_chunk):
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = rs.resample(x)
    x = x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
    if rs.mode == "downsample3d":
        i = idx[0]
        if not first_chunk:                  # original chunk 0 stores x and skips time_conv
            y = rs.time_conv(torch.cat([cache[i], x], dim=2))
        cache[i] = x[:, :, -1:].contiguous()
        idx[0] += 1
        if not first_chunk:
            x = y
    return x


def _middle(mid, x, cache, idx):
    for layer in mid:
        x = _resblock(layer, x, cache, idx) if isinstance(layer, V.ResidualBlock) else _attention(layer, x)
    return x


def _head(head, x, cache, idx):
    for layer in head:
        x = _cconv(layer, x, cache, idx) if isinstance(layer, V.CausalConv3d) else layer(x)
    return x


class StaticDecoder(nn.Module):
    """conv2 + Decoder3d for ONE latent frame; cache in/out as flat tensors."""

    def __init__(self, vae_model):
        super().__init__()
        self.conv2, self.decoder = vae_model.conv2, vae_model.decoder

    def forward(self, z_frame, *flat_cache, first_chunk: bool):
        dec, cache, idx = self.decoder, list(flat_cache), [0]
        x = self.conv2(z_frame)
        x = _cconv(dec.conv1, x, cache, idx)
        x = _middle(dec.middle, x, cache, idx)
        for m in dec.upsamples:
            x = _resblock(m, x, cache, idx) if isinstance(m, V.ResidualBlock) else \
                _resample_dec(m, x, cache, idx, first_chunk)
        x = _head(dec.head, x, cache, idx)
        return (x, *cache)


class StaticEncoder(nn.Module):
    """Encoder3d + conv1 for one chunk (1 frame first, then 4); cache in/out as flat tensors."""

    def __init__(self, vae_model):
        super().__init__()
        self.encoder, self.conv1 = vae_model.encoder, vae_model.conv1

    def forward(self, x, *flat_cache, first_chunk: bool):
        enc, cache, idx = self.encoder, list(flat_cache), [0]
        x = _cconv(enc.conv1, x, cache, idx)
        for m in enc.downsamples:
            x = _resblock(m, x, cache, idx) if isinstance(m, V.ResidualBlock) else \
                _resample_enc(m, x, cache, idx, first_chunk)
        x = _middle(enc.middle, x, cache, idx)
        x = _head(enc.head, x, cache, idx)
        return (self.conv1(x), *cache)


def _dec_cache(vae_model, B, H, W, dtype, device):
    dec, cache = vae_model.decoder, []

    def add(conv, h, w, t=CACHE_T):
        cache.append(torch.zeros(B, conv.in_channels, t, h, w, dtype=dtype, device=device))

    def add_res(blk, h, w):
        for layer in blk.residual:
            if isinstance(layer, V.CausalConv3d):
                add(layer, h, w)

    add(dec.conv1, H, W)
    for m in dec.middle:
        if isinstance(m, V.ResidualBlock):
            add_res(m, H, W)
    for m in dec.upsamples:
        if isinstance(m, V.ResidualBlock):
            add_res(m, H, W)
        else:
            if m.mode == "upsample3d":
                add(m.time_conv, H, W)
            H, W = H * 2, W * 2
    add(dec.head[2], H, W)
    return cache


def _enc_cache(vae_model, B, H, W, dtype, device):
    enc, cache = vae_model.encoder, []

    def add(c, h, w, t=CACHE_T):
        cache.append(torch.zeros(B, c, t, h, w, dtype=dtype, device=device))

    def add_res(blk, h, w):
        for layer in blk.residual:
            if isinstance(layer, V.CausalConv3d):
                add(layer.in_channels, h, w)

    add(enc.conv1.in_channels, H, W)
    for m in enc.downsamples:
        if isinstance(m, V.ResidualBlock):
            add_res(m, H, W)
        else:
            H, W = H // 2, W // 2
            if m.mode == "downsample3d":
                add(m.dim, H, W, t=1)
    for m in enc.middle:
        if isinstance(m, V.ResidualBlock):
            add_res(m, H, W)
    add(enc.head[2].in_channels, H, W)
    return cache


class StaticWanVAE:
    """Drop-in for Wan2_1_VAE.encode / .decode (lists of [C, T, H, W]) with compiled graphs."""

    def __init__(self, vae):            # vae: Wan2_1_VAE with .model on device (bf16)
        install_rewrites()
        self.vae, self.model = vae, vae.model
        n = prepare_nki(self.model)
        if VAE_NKI_ATTN:
            from models.fast_dit import _get_cte
            _get_cte()
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        self.dec = torch.compile(StaticDecoder(self.model), **COMPILE_KW)
        self.enc = torch.compile(StaticEncoder(self.model), **COMPILE_KW)
        logger.info(f"[vae_static21] fullgraph encode/decode, nki_conv_sites={n} nki_attn={VAE_NKI_ATTN}")

    def decode(self, zs):
        out = []
        m, (mean, inv_std) = self.model, self.vae.scale
        for u in zs:
            z = u.unsqueeze(0)
            z = (z / inv_std.view(1, m.z_dim, 1, 1, 1) + mean.view(1, m.z_dim, 1, 1, 1)).to(m.conv2.weight.dtype)
            B, _, T, H, W = z.shape
            cache = _dec_cache(m, B, H, W, z.dtype, z.device)
            frames = []
            for i in range(T):
                res = self.dec(z[:, :, i:i + 1], *cache, first_chunk=(i == 0))
                frames.append(res[0])
                cache = list(res[1:])
            out.append(torch.cat(frames, dim=2).float().clamp_(-1, 1).squeeze(0))
        return out

    def encode(self, videos):
        out = []
        m, (mean, inv_std) = self.model, self.vae.scale
        for u in videos:
            x = u.unsqueeze(0).to(m.conv1.weight.dtype)
            B, _, T, H, W = x.shape
            cache = _enc_cache(m, B, H, W, x.dtype, x.device)
            chunks = [x[:, :, :1]] + [x[:, :, 1 + 4 * i:5 + 4 * i] for i in range((T - 1) // 4)]
            lat = []
            for i, c in enumerate(chunks):
                res = self.enc(c.contiguous(), *cache, first_chunk=(i == 0))
                lat.append(res[0])
                cache = list(res[1:])
            mu = torch.cat(lat, dim=2).chunk(2, dim=1)[0]
            mu = (mu - mean.view(1, m.z_dim, 1, 1, 1)) * inv_std.view(1, m.z_dim, 1, 1, 1)
            out.append(mu.float().squeeze(0))
        return out
