import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# -------------------------
# Helpers: shapes & blocks
# -------------------------

def hw_from_tokens(num_tokens: int, hw: Tuple[int, int] | None = None):
    if hw is not None:
        return hw
    h = w = int(num_tokens ** 0.5)
    assert h * w == num_tokens, f"num_tokens ({num_tokens}) must be a perfect square if hw not provided."
    return h, w

def seq2spatial(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """
    x: (B, N, C) -> (B, H, W, C)
    """
    B, N, C = x.shape
    H, W = hw_from_tokens(N, hw)
    return x.view(B, H, W, C)

def spatial2seq(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, W, C) -> (B, N, C)
    """
    B, H, W, C = x.shape
    return x.view(B, H * W, C)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, dropout=0.0):
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    """
    Thin wrapper around nn.TransformerEncoderLayer to keep API consistent.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=int(mlp_ratio * dim),
            dropout=dropout, activation='gelu', batch_first=True
        )

    def forward(self, x):
        # x: (B, N, C)
        return self.block(x)

class PatchEmbed(nn.Module):
    """
    Conv patchify -> linear embed.
    Produces tokens laid out on (H/ps, W/ps).
    """
    def __init__(self, in_ch, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()   # (B, H', W', C)
        tokens = spatial2seq(x)                  # (B, N, C)
        hw = (H, W)
        return tokens, hw

class PatchMerging(nn.Module):
    """
    Downsample 2x (merge 2x2 tokens) and linearly project to next dim.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.reduction = nn.Linear(4 * in_dim, out_dim)

    def forward(self, x, hw: Tuple[int, int]):
        # x: (B, N, C), hw=(H, W)  -> (B, N/4, C_out), hw'=(H/2,W/2)
        B, N, C = x.shape
        H, W = hw
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for PatchMerging."
        xs = seq2spatial(x, hw)  # (B,H,W,C)

        x00 = xs[:, 0::2, 0::2, :]
        x01 = xs[:, 0::2, 1::2, :]
        x10 = xs[:, 1::2, 0::2, :]
        x11 = xs[:, 1::2, 1::2, :]
        x_cat = torch.cat([x00, x01, x10, x11], dim=-1)  # (B, H/2, W/2, 4C)

        x_red = self.reduction(x_cat)  # (B, H/2, W/2, C_out)
        x_seq = spatial2seq(x_red)     # (B, (H/2)*(W/2), C_out)
        return x_seq, (H // 2, W // 2)

class PatchExpand(nn.Module):
    """
    Upsample 2x in token space using a simple linear + pixel shuffle in (H,W).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Expand channels then rearrange to 2x spatial upsample
        self.proj = nn.Linear(in_dim, 4 * out_dim)

    def forward(self, x, hw: Tuple[int, int]):
        # x: (B, N, C_in), hw=(H,W) -> (B, 4N, C_out), hw'=(2H,2W)
        B, N, C = x.shape
        H, W = hw
        x = self.proj(x)                     # (B,N,4*C_out)
        x = seq2spatial(x, (H, W))           # (B,H,W,4*C_out)

        # pixel-shuffle in (H,W) domain
        B_, H_, W_, CC = x.shape
        C_out = CC // 4
        x = x.view(B_, H_, W_, 2, 2, C_out)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B,H,2,W,2,C)
        x = x.view(B_, H_ * 2, W_ * 2, C_out)         # (B,2H,2W,C)

        x = spatial2seq(x)                 # (B, (2H)*(2W), C_out)
        return x, (H * 2, W * 2)

class UViTEncoderStage(nn.Module):
    """
    One encoder stage: several Transformer blocks, then optional downsample.
    """
    def __init__(self, dim, heads, depth, down_dim=None, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.down = PatchMerging(dim, down_dim) if down_dim is not None else None
        self.pos = nn.Parameter(torch.zeros(1, 1, dim))  # broadcasted 2D-learned-ish; simple & robust

    def forward(self, x, hw):
        # x: (B,N,C)
        x = x + self.pos  # simple learnable pos bias per stage
        for blk in self.blocks:
            x = blk(x)
        skip = x, hw  # save for U-Net skip
        if self.down is not None:
            x, hw = self.down(x, hw)
        return x, hw, skip

class UViTDecoderStage(nn.Module):
    """
    One decoder stage: upsample, fuse skip (concat + linear), then Transformer blocks.
    """
    def __init__(self, dim, heads, depth, up_out_dim, fuse_dim=None, dropout=0.0):
        super().__init__()
        self.up = PatchExpand(dim, up_out_dim)
        fuse_in = up_out_dim + (fuse_dim or up_out_dim)
        self.fuse = nn.Linear(fuse_in, up_out_dim)
        self.blocks = nn.ModuleList([TransformerBlock(up_out_dim, heads, dropout=dropout) for _ in range(depth)])
        self.pos = nn.Parameter(torch.zeros(1, 1, up_out_dim))

    def forward(self, x, hw, skip):
        x, hw = self.up(x, hw)                 # upsample
        skip_x, skip_hw = skip
        assert hw == skip_hw, "Skip spatial size must match after upsample."
        # fuse skip via concat + linear
        x = torch.cat([x, skip_x], dim=-1)
        x = self.fuse(x)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        return x, hw

class UViT(nn.Module):
    """
    U-Net Vision Transformer:
      - PatchEmbed
      - Encoder stages with downsampling
      - Bottleneck
      - Decoder stages with upsampling + skip fusion
      - Patch projection to image space
    """
    def __init__(
        self,
        image_size: int,
        in_ch: int,
        out_ch: int,
        patch_size: int = 8,
        dims: List[int] = (128, 256, 512, 512),
        heads: List[int] = (4, 8, 8, 8),
        depths: List[int] = (2, 2, 2, 2),
        dropout: float = 0.0,
    ):
        """
        dims[i]   = embedding dim at stage i (encoder i / decoder mirrored)
        heads[i]  = attention heads at stage i
        depths[i] = number of Transformer blocks per stage
        """
        super().__init__()
        assert len(dims) == len(heads) == len(depths) >= 2, "Use >=2 stages."
        self.image_size = image_size
        self.patch_size = patch_size
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.out_ch = out_ch

        # Patchify
        self.patch_embed = PatchEmbed(in_ch, dims[0], patch_size)

        # Encoder
        self.enc_stages = nn.ModuleList()
        for i in range(len(dims)):
            down_dim = dims[i + 1] if i + 1 < len(dims) else None
            self.enc_stages.append(
                UViTEncoderStage(dims[i], heads[i], depths[i], down_dim=down_dim, dropout=dropout)
            )

        # Bottleneck (no spatial change)
        bottleneck_dim = dims[-1]
        self.bottleneck = nn.ModuleList([TransformerBlock(bottleneck_dim, heads[-1], dropout=dropout) for _ in range(max(1, depths[-1]))])

        # Decoder (mirror, skipping the bottom-most since it's the bottleneck)
        self.dec_stages = nn.ModuleList()
        for i in reversed(range(len(dims) - 1)):
            in_dim = dims[i + 1]       # current (before upsample)
            up_out_dim = dims[i]       # after upsample target dim
            self.dec_stages.append(
                UViTDecoderStage(
                    dim=in_dim,
                    heads=heads[i],
                    depth=depths[i],
                    up_out_dim=up_out_dim,
                    fuse_dim=dims[i],   # skip feature dim to fuse
                    dropout=dropout
                )
            )

        # Final projection: tokens -> patches -> image
        self.head = nn.Linear(dims[0], patch_size * patch_size * out_ch)

    def forward(self, x):
        """
        x: (B, in_ch, H, W)
        returns: (B, out_ch, H, W)
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size

        # Patchify
        tokens, hw = self.patch_embed(x)   # (B, N, dims[0]), hw=(H/ps,W/ps)

        # Encoder (collect skips)
        skips = []
        hwi = hw
        xi = tokens
        for stage in self.enc_stages:
            xi, hwi, skip = stage(xi, hwi)
            skips.append(skip)
        # xi is deepest features, hwi deepest hw

        # Bottleneck
        for blk in self.bottleneck:
            xi = blk(xi)

        # Decoder (mirror: pop skips; first pop is bottleneck skip we don't use)
        # Remove the last skip which corresponds to deepest stage output *before* downsample
        _ = skips.pop()  # not used (matches bottleneck resolution)
        for dec in self.dec_stages:
            skip = skips.pop()  # (x_skip, hw_skip)
            xi, hwi = dec(xi, hwi, skip)

        # Project tokens -> patches -> image
        # xi is at stage 0 resolution (H/ps, W/ps)
        B_, N_, C0 = xi.shape
        patches = self.head(xi)  # (B, N, ps*ps*out_ch)
        ps = self.patch_size
        out = patches.view(B_, int(N_**0.5), int(N_**0.5), out.size(-1) if (out := patches) is not None else patches.size(-1))
        # (B, H/ps, W/ps, ps*ps*out_ch)
        out = out.view(B_, int(N_**0.5), int(N_**0.5), self.out_ch, ps, ps)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, out_ch, H/ps, ps, W/ps, ps)
        out = out.view(B_, self.out_ch, self.image_size, self.image_size)
        return out