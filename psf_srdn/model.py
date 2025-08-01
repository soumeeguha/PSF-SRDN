import math
from inspect import isfunction
from functools import partial

from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

# https://github.com/huggingface/blog/blob/main/annotated-diffusion.md

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        # print('Residual')
        # print(x.size())
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)
    
class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, timesteps, *, time_emb_dim=112, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        time_embed_input = timesteps*time_emb_dim

        # print('Time_embed_dim: ', time_emb_dim)
        ########
        self.time_embed_net = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embed_input, time_emb_dim),
            # nn.SiLU(), nn.Linear(time_embed_input//4, time_embed_input//8),
            # nn.SiLU(), nn.Linear(time_embed_input//8, time_emb_dim)
        )
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        # print('Convnext')
        h = self.ds_conv(x)
        # print('h: ', h.size())

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            # print('time_emb: ', time_emb.size(), time_emb.flatten().unsqueeze(0).size())
            time_emb = self.time_embed_net(time_emb.flatten()).unsqueeze(0).repeat(x.size()[0], 1)
            # print('time_emb: ', time_emb.size())
            condition = self.mlp(time_emb)
            # print('condition: ', condition.size(), condition.flatten().size())
            # print(rearrange(condition, "b c -> b c 1 1").size())
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        # print('Prenorm: ', x.size())
        x = self.norm(x)
        return self.fn(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim1, dim2, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q1 = nn.Conv2d(dim1, hidden_dim, 1, bias=False)
        self.to_kv2 = nn.Conv2d(2, hidden_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim1, 1)

    def forward(self, x1, x2):
        b, c1, h1, w1 = x1.shape
        b, c2, h2, w2 = x2.shape
        
        if (h1, w1) != (h2, w2):
            x2 = F.interpolate(x2, size=(h1, w1), mode='bilinear', align_corners=False)
        
        q1 = self.to_q1(x1)
        k2, v2 = self.to_kv2(x2.squeeze().unsqueeze(0)).chunk(2, dim=1)
        
        batch_size_q, channels, height, width = q1.shape
        batch_size_kv, channels, height, width = k2.shape

        d_k = channels  # dimension of keys

        k2 = k2.repeat(batch_size_q // batch_size_kv, 1, 1, 1)  
        v2 = v2.repeat(batch_size_q // batch_size_kv, 1, 1, 1)  

        # Reshape q, k, and v to prepare for attention computation
        q1 = q1.view(batch_size_q, channels, -1)  
        k2 = k2.view(batch_size_q, channels, -1) 
        v2 = v2.view(batch_size_q, channels, -1) 

        # Step 3: Compute the dot-product attention (q @ k^T) / sqrt(d_k)
        k_t = k2.transpose(-2, -1)  # [128, 4096, 256]
        scores = torch.bmm(q1, k_t) / (d_k ** 0.5) 

        # Step 4: Apply softmax to obtain attention weights
        attention_weights = torch.softmax(scores, dim=-1)  

        # Step 5: Multiply attention weights by v
        out = torch.bmm(attention_weights, v2) 
        # Step 6: Reshape the output back to the original spatial dimensions
        out = out.view(batch_size_q, channels, height, width) 
     
        return self.to_out(out) + x1
    

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        psf_dim,
        timesteps,
        init_dim=None,
        out_dim=None, 
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        with_psf_emb = True,
        resnet_block_groups=8,
        attn_heads = 8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # time embeddings
        if with_psf_emb:
            # psf_dim = dim * 2
            self.psf_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(psf_dim*2),
                nn.Linear(psf_dim*2, psf_dim),
                nn.GELU(),
                nn.Linear(psf_dim, psf_dim),
            )
        else:
            psf_dim = None
            self.psf_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [   ####### time_dim_in ---> batch_size
                        block_klass(dim_in, dim_out, timesteps, time_emb_dim=time_dim), #block1
                        block_klass(dim_out, dim_out, timesteps, time_emb_dim=time_dim), #block2
                        # Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, dim_out, heads = attn_heads))), #psf_attn
                        Residual(SpatialLinearAttention(dim_out, psf_dim, heads = attn_heads)), #psf_attn
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))), #attn
                        Downsample(dim_out) if not is_last else nn.Identity(), #downsample
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, timesteps, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, timesteps, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, timesteps, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, timesteps, time_emb_dim=time_dim),
                        # Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))), #psf_attn
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim, timesteps), nn.Conv2d(dim, out_dim, 1)
        )

        self.sigmoid_final = nn.Sigmoid()

    def forward(self, x, time, sigma_xy):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        p = self.psf_mlp(sigma_xy) if exists(self.psf_mlp) else None

        h = []

        # downsample
        for block1, block2, spatial_attn, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x, p.unsqueeze(1))
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.sigmoid_final(self.final_conv(x))
    
