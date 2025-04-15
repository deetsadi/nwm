# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    # scale_, shift_ = scale, shift
    # if len(shift.shape) == 2:
    #     scale_ = scale.unsqueeze(1)
    #     shift_ = shift.unsqueeze(1)

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ActionEmbedderMLP(nn.Module):
    """
    Embeds action into vector representations.
    """
    def __init__(self, d_action, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_action, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, actions):
        return self.mlp(torch.mean(actions, dim=1))
    
# class ActionEmbedder(nn.Module):
#     """
#     Embeds action into vector representations.
#     """
#     def __init__(self, d_action, d_model, n_heads=4, n_layers=2, seq_len=60):
#         super().__init__()
#         self.embedding = nn.Linear(d_action, d_model)
#         self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))
#         enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
#         self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

#         # self.mlp = nn.Sequential(
#         #     nn.Linear(d_model, 2 * d_model, bias=True),
#         #     nn.GELU(),
#         #     nn.Linear(2 * d_model, d_model, bias=True),
#         # )

#     def forward(self, actions):
#         B, T, _ = actions.shape
#         x = self.embedding(actions) + self.pos_emb
#         cls = self.cls_token.expand(B, -1, -1)
#         x = torch.cat([cls, x], dim=1)
#         x = self.enc(x)

#         return x[:, 0]

class ActionEmbedderTransformer(nn.Module):
    """
    Embeds action into vector representations.
    """
    def __init__(self, d_action, d_model, n_heads=8, n_layers=8, max_seq_len=100, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Linear(d_action, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.proj = nn.Linear(hidden_dim, d_model)

        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model, 2 * d_model, bias=True),
        #     nn.GELU(),
        #     nn.Linear(2 * d_model, d_model, bias=True),
        # )

    def forward(self, actions):
        B, T, _ = actions.shape

        action_mask = (actions == -100).any(dim=-1)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=actions.device)
        final_mask = torch.cat([cls_mask, action_mask], dim=1)
        src_key_padding_mask = final_mask

        x = self.embedding(actions) + self.pos_emb[:, :actions.size(1), :]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.enc(x, src_key_padding_mask=src_key_padding_mask)

        return self.proj(x[:, 0])

class ActionEmbedder(nn.Module):
    def __init__(self, d_action, d_model, embedder_type='mlp'):
        super().__init__()
        if embedder_type == 'transformer':
            self.embedder = ActionEmbedderTransformer(d_action, d_model)
        else:
            self.embedder = ActionEmbedderMLP(d_action, d_model)
    
    def forward(self, actions):
        return self.embedder(actions)

class ActionEncoder(nn.Module):
    """
    Embeds action into vector representations.
    """
    def __init__(self, d_action, d_model, n_heads=8, n_layers=8, max_seq_len=100, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Linear(d_action, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.proj = nn.Linear(hidden_dim, d_model)

    def forward(self, actions):
        B, T, _ = actions.shape

        action_mask = (actions == -100).any(dim=-1)
        src_key_padding_mask = action_mask

        x = self.embedding(actions) + self.pos_emb[:, :actions.size(1), :]
        x = self.enc(x, src_key_padding_mask=src_key_padding_mask)

        return self.proj(x)

#################################################################################
#                                 Core CDiT Model                                #
#################################################################################

# class CDiTBlock(nn.Module):
#     """
#     A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 11 * hidden_size, bias=True)
#         )
#         # self.adaLN_modulation_goal = nn.Sequential(
#         #     nn.SiLU(),
#         #     nn.Linear(hidden_size, 9 * hidden_size, bias=True)
#         # )
#         # self.adaLN_modulation_context = nn.Sequential(
#         #     nn.SiLU(),
#         #     nn.Linear(hidden_size, 2 * hidden_size, bias=True)
#         # )

#         self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

#     def forward(self, x, c, x_cond):
#         shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=-1)
#         x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
#         x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
#         x = x + gate_ca_x.unsqueeze(1) * self.cttn(query=modulate(self.norm2(x), shift_ca_x, scale_ca_x), key=x_cond_norm, value=x_cond_norm, need_weights=False)[0]
#         x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
#         return x

class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, act_block = True, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )
        # self.adaLN_modulation_goal = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        # )
        # self.adaLN_modulation_context = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        # )
        if act_block:
            self.cttn_act = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c, x_cond, act=None):
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(query=modulate(self.norm2(x), shift_ca_x, scale_ca_x), key=x_cond_norm, value=x_cond_norm, need_weights=False)[0]

        if act is not None:
            act_mask = (act == -100).any(dim=-1)  # (B, T_act)
            x = x + self.cttn_act(self.norm3(x), act, act, key_padding_mask=act_mask)[0]  # cross-attn with mask

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm4(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=2,
        in_channels=4,
        action_size=25,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        act_block=True
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(action_size, hidden_size)

        self.act_block = act_block
        if self.act_block:
            self.action_encoder = ActionEncoder(action_size, hidden_size)
        num_patches = self.x_embedder.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(self.context_size + 1, num_patches, hidden_size), requires_grad=True) # for context and for predicted frame
        # self.pos_embed_act = nn.Parameter(torch.zeros(self.context_size, action_size), requires_grad=True)

        self.blocks = nn.ModuleList([CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, act_block=act_block) for _ in range(depth)])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        nn.init.normal_(self.pos_embed, std=0.02)
        # nn.init.normal_(self.pos_embed_act, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Initialize action embedding:
        if self.act_block:
            nn.init.normal_(self.y_embedder.embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.embedder.mlp[2].weight, std=0.02)
        else:
            nn.init.normal_(self.y_embedder.embedder.pos_emb, std=0.02)
            nn.init.normal_(self.y_embedder.embedder.cls_token, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
            
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            # nn.init.constant_(block.adaLN_modulation_goal[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation_goal[-1].bias, 0)
            # nn.init.constant_(block.adaLN_modulation_context[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation_context[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, x_cond, rel_t):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed[self.context_size:]
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1])) + self.pos_embed[:self.context_size]  # (N, T, D), where T = H * W / patch_size ** 2.flatten(1, 2)
        x_cond = x_cond.flatten(1, 2)
        t = self.t_embedder(t[..., None])
        y_pooled = self.y_embedder(y)
        time_emb = self.time_embedder(rel_t[..., None])

        c = t + time_emb + y_pooled # if training on unlabeled data, dont add y.

        if self.act_block:
            y = self.action_encoder(y)

        for block in self.blocks:
            x = block(x, c, x_cond, act=y)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                  #
#################################################################################

def CDiT_XL_2(**kwargs):
    return CDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


CDiT_models = {
    'CDiT-XL/2': CDiT_XL_2, 
    'CDiT-L/2':  CDiT_L_2, 
    'CDiT-B/2':  CDiT_B_2, 
    'CDiT-S/2':  CDiT_S_2
}
