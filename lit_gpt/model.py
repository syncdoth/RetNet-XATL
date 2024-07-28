"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from flash_attn.ops.fused_dense import FusedDense, FusedMLP
from mamba_ssm import Mamba
from .rmsnorm import RMSNormNoWeight
from .fused_rotary_embedding import apply_rotary_emb_func
from .fused_parallel_retention import parallel_retention as fused_parallel_retention
try:
    from fla.ops.retention.chunk_fuse import fused_chunk_retention
except ModuleNotFoundError:
    fused_chunk_retention = None
# ATTENTIONS  # TODO
# from .based import BasedLinearAttention
# from .linformer import LinformerSelfAttention
# from .aft_attention import AFTFullAttention
# from .linear_attention import LinearAttention
# from .reformer_attention import ReformerAttention

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class GPT(nn.Module):

    def __init__(self, config: Config, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.eos_token_id = config.eos_token_id
        self.dtype = dtype

        self.lm_head = FusedDense(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        ### added from retnet ####
        if config.layernorm_embedding:
            self.ln_e = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(config.n_embd)
        ##########################
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            ))
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,
                                  mean=0.0,
                                  std=math.sqrt(2.0 / 5 / self.config.n_embd))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, (nn.Linear, FusedDense)):
            torch.nn.init.normal_(module.weight,
                                  mean=0.0,
                                  std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (
                    name == "w3.weight" and isinstance(module, SwiGLU) or
                (name == "proj.weight" and isinstance(module,
                                                      (CausalSelfAttention, MultiScaleRetention)))
            ):  #if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / n_layer)

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def forward(self,
                idx: torch.Tensor,
                max_seq_length: Optional[int] = None,
                input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        # NOTE: commented out because fused_parallel_retention does masking within the function
        # if self.config.use_retention and (self.mask_cache is None or self.config.use_bda):
        #     self.mask_cache = build_retention_mask_cache(self.config,
        #                                                  max_seq_length,
        #                                                  idx,
        #                                                  self.dtype,
        #                                                  use_bda=self.config.use_bda,
        #                                                  eos_token_id=self.eos_token_id)
        if (use_kv_cache and self.mask_cache is None) or self.config.use_bda:
            self.mask_cache = self.build_mask_cache(idx, use_bda=self.config.use_bda)

        cos, sin = self.rope_cache
        if use_kv_cache:

            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            cos = cos[:T]
            sin = sin[:T]
            # NOTE: commented out because fused_parallel_retention does masking within the function
            # if self.config.use_retention or self.config.use_bda:
            if self.config.use_bda:
                mask = self.mask_cache
            else:
                mask = None

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = x * self.embed_scale
        if self.config.layernorm_embedding:
            x = self.ln_e(x)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, (cos, sin), max_seq_length, mask)
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length,
                                                                    cos.size(-1) * 2)
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, (cos, sin), max_seq_length, mask, input_pos,
                                             self.kv_caches[i])

        x = self.transformer.ln_f(x)

        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=self.dtype,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
            retnet_compat=self.config.use_retention,
        )

    def build_mask_cache(self, idx: torch.Tensor, use_bda: bool = False) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size),
                          device=idx.device,
                          dtype=torch.bool)
        mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
        if use_bda:
            mask = mask.repeat(idx.size(0), 1, 1, 1)
            for b_i, _idx in enumerate(idx):
                eos_positions = torch.where(_idx == self.eos_token_id)[0]
                for pos in eos_positions:
                    mask[b_i, 0][pos + 1:, :pos + 1] = False

        return mask

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int,
                        rope_cache_length: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups

        k_cache_shape = (
            B,
            max_seq_length,
            heads,
            rope_cache_length + self.config.head_size -
            int(self.config.rotary_percentage * self.config.head_size),
        )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [(torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape,
                                                                        device=device))
                for _ in range(self.config.n_layer)]


class Block(nn.Module):

    def __init__(self, config: Config, layer_i: int) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)

        if config.hybrid_attention_layers is not None and layer_i in config.hybrid_attention_layers:
            self.attn = CausalSelfAttention(config)
        elif config.use_retention:
            self.attn = MultiScaleRetention(config)
        elif config.use_mamba:
            self.attn = Mamba(d_model=config.n_embd,
                              d_state=config.mamba_d_state,
                              d_conv=config.mamba_d_conv,
                              expand=config.mamba_expand)
        elif config.linear_attention is not None:
            raise NotImplementedError
            # if config.linear_attention == "based":
            #     self.attn = BasedLinearAttention(config)
            # elif config.linear_attention == "linformer":
            #     self.attn = LinformerSelfAttention(config)
            # elif config.linear_attention == "aft":
            #     self.attn = AFTFullAttention(config)
            # elif config.linear_attention == "linear":
            #     self.attn = LinearAttention(config)
            # elif config.linear_attention == "reformer":
            #     self.attn = ReformerAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        n_1 = self.norm_1(x)
        if not self.config.use_mamba:
            h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)
        else:
            h = self.attn(n_1)
            new_kv_cache = None
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm).")

            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = FusedDense(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = FusedDense(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv,
                       self.config.head_size)  # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.config.head_size)
        v = v.reshape(B, T, -1, self.config.head_size)

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        # n_elem = int(self.config.rotary_percentage * self.config.head_size)

        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(self,
                                     q: torch.Tensor,
                                     k: torch.Tensor,
                                     v: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None):
        scale = 1.0 / math.sqrt(self.config.head_size)

        if (FlashAttention2Available and mask is None and q.device.type == "cuda" and
                q.dtype in (torch.float16, torch.bfloat16)):
            from flash_attn import flash_attn_func

            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(q,
                                                             k,
                                                             v,
                                                             attn_mask=mask,
                                                             dropout_p=0.0,
                                                             scale=scale,
                                                             is_causal=mask is None)
        # no sdpa
        # qk_mat = q @ k.transpose(-1, -2) * scale
        # attn_weights = nn.functional.softmax(qk_mat, dim=-1)
        # y = torch.matmul(attn_weights, v)
        return y.transpose(1, 2)


class MultiScaleRetention(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.reten = FusedDense(config.n_embd, shape, bias=config.bias)
        # TODO: this is wrong, but Retnet-410M and 1B models already being trained with this.
        if config.use_ln_for_groupnorm:
            self.group_norm = config.norm_class(config.head_size, eps=config.norm_eps)
        else:
            self.group_norm = RMSNormNoWeight(config.head_size, eps=config.norm_eps)

        if config.retnet_bottleneck:
            self.down_proj = FusedDense(config.n_embd, config.retnet_bottleneck, bias=config.bias)
            self.gate_proj = FusedDense(config.n_embd, config.retnet_bottleneck, bias=config.bias)
            self.proj = FusedDense(config.retnet_bottleneck, config.n_embd, bias=config.bias)
        else:
            self.down_proj = None
            self.gate_proj = FusedDense(config.n_embd, config.n_embd, bias=config.bias)
            self.proj = FusedDense(config.n_embd, config.n_embd, bias=config.bias)

        self.gate_fn = torch.nn.functional.silu
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        reference forward function from flash-linear-attention.

        def forward(self, x):
            mode = self.mode
            q1 = rearrange(self.q_proj(x), '... (h d) -> ... h d', h=self.num_heads)
            k1 = rearrange(self.k_proj(x), '... (h d) -> ... h d', h=self.num_heads)
            q, k = self.rotary(q1, k1)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
            if mode == 'chunk':
                o = chunk_retention(q, k, v)
            elif mode == 'fused_chunk':
                o = fused_chunk_retention(q, k, v)
            elif mode == 'parallel':
                o = parallel_retention(q, k, v)
            elif mode == 'fused_recurrent':
                o = fused_recurrent_retention(q, k, v)
            else:
                raise NotImplementedError

            o = rearrange(o, 'b h l d -> b l h d')
            g = self.g_proj(x)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b l h d -> b l (h d)')

            else:
                o = self.g_norm(o)
                o = rearrange(o, 'b l h d -> b l (h d)')
                o = o * self.gate_fn(g)
            o = self.o_proj(o)
            return o
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.reten(x)
        g = self.gate_fn(self.gate_proj(x))

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv,
                       self.config.head_size)  # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.config.head_size)
        v = v.reshape(B, T, -1, self.config.head_size)

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, True, True)
        k = apply_rotary_emb_func(k, cos, sin, True, True)

        # n_elem = int(self.config.rotary_percentage * self.config.head_size)

        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        q = q.transpose(1, 2)  # (b, nh, t, hs)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.config.use_chunkwise_retention:
            y, _ = fused_chunk_retention(q, k, v)
        else:
            y = self.parallel_retention(q, k, v, mask)
        y = y.transpose(1, 2)  # (b, t, nh, hs)

        y = self.group_norm(y)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        if self.down_proj is not None:
            y = self.down_proj(y)
        y = self.proj(g * y)

        return y, kv_cache

    def parallel_retention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           mask: torch.Tensor):
        scale = 1.0 / math.sqrt(self.config.head_size)

        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

        if fused_parallel_retention is not None:
            return fused_parallel_retention(q, k, v)  # NOTE: the fused function from fla

        qk_mat = q @ k.transpose(-1, -2) * scale
        qk_mat = qk_mat * mask.to(qk_mat.dtype)

        # qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        y = qk_mat @ v
        return y


class GptNeoxMLP(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        # self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.mlp = FusedMLP(config.n_embd,
                            config.intermediate_size,
                            config.n_embd,
                            bias1=config.bias,
                            bias2=config.bias,
                            activation="gelu_approx")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.fc(x)
        # x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        # return self.proj(x)
        return self.mlp(x)


class LLaMAMLP(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        # self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.swiglu = SwiGLU(config.n_embd,
                             config.intermediate_size,
                             bias=False,
                             _pack_weights=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_fc_1 = self.fc_1(x)
        # x_fc_2 = self.fc_2(x)
        # x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        # return self.proj(x)
        return self.swiglu(x)


def build_rope_cache(seq_len: int,
                     n_elem: int,
                     dtype: torch.dtype,
                     device: torch.device,
                     base: int = 10000,
                     condense_ratio: int = 1,
                     retnet_compat: bool = False) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    if retnet_compat:
        theta = 1.0 / (base**torch.linspace(0, 1, n_elem // 2, device=device))
    else:
        theta = 1.0 / (base**(torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., :head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2:]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)


def build_retention_mask_cache(config,
                               slen,
                               idx,
                               dtype,
                               use_bda: bool = False,
                               eos_token_id: int = 2):
    # decay (gamma)
    if config.use_lm_decay:
        # NOTE: alternative way described in the paper
        s = torch.log2(torch.tensor(1 / 32))
        e = torch.log2(torch.tensor(1 / 512))
        decay = torch.log2(1 - torch.exp(torch.linspace(s, e, config.n_head)))  # [h,]
    else:
        decay = torch.log2(1 - 2.**(-5. - torch.arange(config.n_head, dtype=torch.float)))

    index = torch.arange(slen).to(decay)

    mask = torch.tril(torch.ones(slen, slen)).to(decay)

    if use_bda:
        mask = mask.unsqueeze(0).repeat(idx.size(0), 1, 1)  # [B, t, t]
        for b_i, _idx in enumerate(idx):
            eos_positions = torch.where(_idx == eos_token_id)[0]
            for pos in eos_positions:
                mask[b_i][pos + 1:, :pos + 1] = 0.0
        # [B, t, t]
        mask = torch.masked_fill(index[None, :, None] - index[None, None, :], ~mask.bool(),
                                 float("inf"))
        mask = torch.exp(mask.unsqueeze(1) * decay[None, :, None, None])  # [B, h, t, t]
    else:
        # [t, t]
        mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
        mask = torch.exp2(mask * decay[:, None, None])  # [h, t, t]
        mask = mask.unsqueeze(0)  # [1, h, t, t]
    mask = torch.nan_to_num(mask)

    # scaling
    # mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
    # mask = torch.nan_to_num(mask, nan=0.0).to(idx.device)

    if dtype == torch.bfloat16:
        return mask.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return mask.half()
    return mask
