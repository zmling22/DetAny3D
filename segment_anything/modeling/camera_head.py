
import torch
import torch.nn as nn


from functools import partial
import torch.nn.functional as F
from einops import rearrange

def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def divisible_by(numer, denom):
    return (numer % denom) == 0


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        gated: bool = False,
        output_dim = None,
    ):
        super().__init__()
        if gated:
            expansion = int(expansion * 2 / 3)
        hidden_dim = int(input_dim * expansion)
        output_dim = default(output_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU() if not gated else SwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.dropout(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = context_dim or dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv = nn.Linear(context_dim, dim * 2)
        self.q = nn.Linear(dim, dim)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()

    def attn(
        self,
        x: torch.Tensor,
        attn_bias = None,
        context = None,
        pos_embed = None,
        pos_embed_context = None,
        rope = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b h n d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b h n d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=attn_bias
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_bias = None,
        context = None,
        pos_embed = None,
        pos_embed_context = None,
        rope = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = (
            self.ls1(
                self.attn(
                    x,
                    rope=rope,
                    attn_bias=attn_bias,
                    context=context,
                    pos_embed=pos_embed,
                    pos_embed_context=pos_embed_context,
                )
            )
            + x
        )
        x = self.ls2(self.mlp(x)) + x
        return x


class CameraHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.aggregate1 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.aggregate2 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.latents_pos = nn.Parameter(
            torch.randn(1, 4, hidden_dim), requires_grad=True
        )
        self.in_features = MLP(hidden_dim, expansion=2, dropout=dropout)
        self.project_cls = MLP(hidden_dim, dropout=dropout)
        self.out = MLP(hidden_dim, expansion=2, dropout=0.0, output_dim=1)

    def fill_intrinsics(self, x):
        camera_intrinsics = torch.zeros(
            x.shape[0], 3, 3, device=x.device, requires_grad=False
        )
        camera_intrinsics[:, 0, 0] = x[:, 0].exp()
        camera_intrinsics[:, 1, 1] = x[:, 1].exp()
        camera_intrinsics[:, 0, 2] = x[:, 2].sigmoid()
        camera_intrinsics[:, 1, 2] = x[:, 3].sigmoid()
        camera_intrinsics[:, 2, 2] = 1.0
        return camera_intrinsics

    def forward(self, features, cls_tokens, pos_embed) -> torch.Tensor:
        # import ipdb;ipdb.set_trace()
        features = features.unbind(dim=-1)
        cls_tokens = self.project_cls(cls_tokens)
        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)
        features = self.in_features(torch.cat(features, dim=1) + pos_embed)
        features = torch.cat((features, cls_tokens), dim=1)
        cls_tokens = self.aggregate1(
            cls_tokens, context=features, pos_embed=latents_pos
        )
        cls_tokens = self.aggregate2(
            cls_tokens, context=features, pos_embed=latents_pos
        )

        # project to intrinsics
        x = self.out(cls_tokens).squeeze(-1)
        camera_intrinsics = self.fill_intrinsics(x)

        return camera_intrinsics

    def set_shapes(self, shapes):
        self.shapes = shapes