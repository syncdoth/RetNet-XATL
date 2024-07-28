from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

import lit_gpt.model
from lit_gpt.utils import find_multiple


@dataclass
class Config:
    org: str = "Lightning-AI"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    gelu_approximate: str = "none"
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1
    ### retnet ###
    use_retention: bool = False
    layernorm_embedding: bool = False
    use_lm_decay: bool = False  # TODO: need to set to False when using fused MSR
    no_scale_embedding: bool = True
    use_ln_for_groupnorm: bool = False  # NOTE: required for old, wrong models
    retnet_bottleneck: int = None  # use down_proj for the gating in retention
    use_chunkwise_retention: bool = False  # TODO: experimental
    ### Block-Diagonal-Attention ###
    use_bda: bool = False
    _eos_token_id: int = None
    ### other attentions ###
    linear_attention: str = None

    ### Hybrid ###
    hybrid_attention_layers: list[int] = None
    hybrid_att_window: int = None

    ### Mamba ###
    use_mamba: bool = False
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: float = 2

    def __post_init__(self):
        # error checking
        assert self.n_embd % self.n_head == 0
        if self.use_retention:
            assert self.linear_attention is None, f"using linear attention={self.linear_attention} but also have use_retention=True"
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(lit_gpt.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from lit_gpt.rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from lit_gpt.rmsnorm import FusedRMSNorm
            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)

    @property
    def eos_token_id(self) -> int:
        if self._eos_token_id is None:
            if self.use_bda:
                raise ValueError("The config needs to set the `eos_token_id` when using BDA.")
            print("Warning: `eos_token_id` is not set.")
            return None
        return self._eos_token_id


########################
# Stability AI StableLM
########################
configs = [
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json
    dict(org="stabilityai", name="stablelm-base-alpha-3b", padding_multiple=512),
    # https://huggingface.co/stabilityai/stablelm-base-alpha-7b/blob/main/config.json
    dict(org="stabilityai",
         name="stablelm-base-alpha-7b",
         n_head=48,
         n_embd=6144,
         padding_multiple=256),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/blob/main/config.json
    dict(org="stabilityai", name="stablelm-tuned-alpha-3b", n_head=32, padding_multiple=512),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b/blob/main/config.json
    dict(org="stabilityai",
         name="stablelm-tuned-alpha-7b",
         n_head=48,
         n_embd=6144,
         padding_multiple=256),
]

####################
# EleutherAI Pythia
####################
pythia = [
    # https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
    dict(org="EleutherAI",
         name="pythia-70m",
         block_size=2048,
         n_layer=6,
         n_embd=512,
         n_head=8,
         padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json
    dict(org="EleutherAI",
         name="pythia-160m",
         block_size=2048,
         n_layer=12,
         n_embd=768,
         n_head=12,
         padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json
    dict(org="EleutherAI",
         name="pythia-410m",
         block_size=2048,
         n_layer=24,
         n_embd=1024,
         n_head=16,
         padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-1b/blob/main/config.json
    dict(org="EleutherAI",
         name="pythia-1b",
         block_size=2048,
         n_layer=16,
         n_embd=2048,
         n_head=8,
         padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-1.4b/blob/main/config.json
    dict(org="EleutherAI",
         name="pythia-1.4b",
         block_size=2048,
         n_layer=24,
         n_embd=2048,
         n_head=16,
         padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json
    dict(org="EleutherAI",
         name="pythia-2.8b",
         block_size=2048,
         n_layer=32,
         n_embd=2560,
         n_head=32,
         padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-6.9b/blob/main/config.json
    dict(org="EleutherAI",
         name="pythia-6.9b",
         block_size=2048,
         n_layer=32,
         n_embd=4096,
         n_head=32,
         padding_multiple=256),
    # https://huggingface.co/EleutherAI/pythia-12b/blob/main/config.json
    dict(org="EleutherAI",
         name="pythia-12b",
         block_size=2048,
         n_layer=36,
         n_embd=5120,
         n_head=40,
         padding_multiple=512),
]
configs.extend(pythia)
for c in pythia:
    copy = c.copy()
    copy["name"] = f"{c['name']}-deduped"
    configs.append(copy)

####################################
# togethercomputer RedPajama INCITE
####################################
redpajama_incite = [
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1/blob/main/config.json
    dict(
        org="togethercomputer",
        name="RedPajama-INCITE-{}-3B-v1",
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        n_head=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base/blob/main/config.json
    dict(
        org="togethercomputer",
        name="RedPajama-INCITE-7B-{}",
        block_size=2048,
        n_layer=32,
        n_embd=4096,
        n_head=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # this redirects to the checkpoint above. kept for those who had the old weights already downloaded
    dict(
        org="togethercomputer",
        name="RedPajama-INCITE-{}-7B-v0.1",
        block_size=2048,
        n_layer=32,
        n_embd=4096,
        n_head=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
]
for c in redpajama_incite:
    for kind in ("Base", "Chat", "Instruct"):
        copy = c.copy()
        copy["name"] = c["name"].format(kind)
        configs.append(copy)

#################
# TII UAE Falcon
#################
falcon = [
    # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
    dict(
        org="tiiuae",
        name="falcon-7b{}",
        block_size=2048,
        padded_vocab_size=65024,
        n_layer=32,
        n_head=71,
        n_embd=4544,
        rotary_percentage=1.0,
        parallel_residual=True,
        n_query_groups=1,
        bias=False,
        # this is not in the config, but in the original model implementation, only for this config
        shared_attention_norm=True,
    ),
    # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
    dict(
        org="tiiuae",
        name="falcon-40b{}",
        block_size=2048,
        padded_vocab_size=65024,
        n_layer=60,
        n_head=128,
        n_embd=8192,
        rotary_percentage=1.0,
        parallel_residual=True,
        n_query_groups=8,
        bias=False,
    ),
]
for c in falcon:
    for kind in ("", "-instruct"):
        copy = c.copy()
        copy["name"] = c["name"].format(kind)
        configs.append(copy)

#############################
# StatNLP Research
#############################
tiny_LLaMA = [
    # https://twitter.com/cwolferesearch/status/1691929174175264858
    dict(
        org="StatNLP-research",
        name="tiny_LLaMA_1b",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=32,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  #Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=4,
    ),
    dict(
        org="StatNLP-research",
        name="tiny_LLaMA_120M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embd=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        n_query_groups=1,
    ),
    dict(
        org="StatNLP-research",
        name="code_tiny_LLaMA_1b",
        block_size=8192,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=32,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  #Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=4,
        condense_ratio=4),
]
configs.extend(tiny_LLaMA)

#############################
# OpenLM Research Open LLaMA
#############################
open_LLaMA = [
    # https://huggingface.co/openlm-research/open_llama_3b/blob/main/config.json
    dict(
        org="openlm-research",
        name="open_llama_3b",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=26,
        n_head=32,
        n_embd=3200,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=8640,
    ),
    # https://huggingface.co/openlm-research/open_llama_7b/blob/main/config.json
    dict(
        org="openlm-research",
        name="open_llama_7b",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/openlm-research/open_llama_13b/blob/main/config.json
    dict(
        org="openlm-research",
        name="open_llama_13b",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
]
configs.extend(open_LLaMA)

###############
# LMSYS Vicuna
###############
vicuna = [
    # https://huggingface.co/lmsys/vicuna-7b-v1.3/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-7b-v1.3",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/lmsys/vicuna-13b-v1.3/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-13b-v1.3",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/lmsys/vicuna-33b-v1.3/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-33b-v1.3",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=60,
        n_head=52,
        n_embd=6656,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=17920,
    ),
    dict(
        org="lmsys",
        name="vicuna-7b-v1.5",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    dict(
        org="lmsys",
        name="vicuna-7b-v1.5-16k",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        condense_ratio=4,
    ),
    dict(
        org="lmsys",
        name="vicuna-13b-v1.5",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    dict(
        org="lmsys",
        name="vicuna-13b-v1.5-16k",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        condense_ratio=4,
    ),
]
configs.extend(vicuna)

#################
# LMSYS LongChat
#################
long_chat = [
    # https://huggingface.co/lmsys/longchat-7b-16k/blob/main/config.json
    dict(
        org="lmsys",
        name="longchat-7b-16k",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        condense_ratio=8,
    ),
    # https://huggingface.co/lmsys/longchat-13b-16k/blob/main/config.json
    dict(
        org="lmsys",
        name="longchat-13b-16k",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        condense_ratio=8,
    ),
]
configs.extend(long_chat)

######################
# NousResearch Hermes
######################
nous_research = [
    # https://huggingface.co/NousResearch/Nous-Hermes-13B/blob/main/config.json
    dict(
        org="NousResearch",
        name="Nous-Hermes-13b",
        block_size=2048,
        padded_vocab_size=32001,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    )
]
configs.extend(nous_research)

###############
# Meta LLaMA 2
###############
llama_2 = [
    # https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
    dict(
        org="meta-llama",
        name="Llama-2-7b{}-hf",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    dict(
        org="meta-llama",
        name="CodeLlama-2-7b-hf",
        block_size=4096,
        vocab_size=32016,
        padded_vocab_size=32016,
        padding_multiple=64,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
    dict(
        org="meta-llama",
        name="Llama-2-13b{}-hf",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
    dict(
        org="meta-llama",
        name="Llama-2-70b{}-hf",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    ),
]
for c in llama_2:
    for kind in ("", "-chat"):
        copy = c.copy()
        copy["name"] = c["name"].format(kind)
        configs.append(copy)

##########################
# Stability AI FreeWilly2
##########################
freewilly_2 = [
    # https://huggingface.co/stabilityai/FreeWilly2/blob/main/config.json
    dict(
        org="stabilityai",
        name="FreeWilly2",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    )
]
configs.extend(freewilly_2)

################
# Microsoft Phi
################
phi = [
    # https://huggingface.co/microsoft/phi-1_5/blob/main/config.json
    dict(
        name="phi-1_5",
        vocab_size=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_embd=2048,
        n_layer=24,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
]
configs.extend(phi)

###############
# RetNet
###############
retnet = [
    dict(
        org="syncdoth",
        name="RetNet-1B",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=24,
        n_head=8,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=3456,
        use_retention=True,
        layernorm_embedding=True,
        use_lm_decay=False,
        no_scale_embedding=False,
    ),
    dict(
        org="NucleusAI",
        name="NucleusX-1B",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=8,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        use_retention=True,
        layernorm_embedding=True,
        use_lm_decay=True,
        no_scale_embedding=False,
    ),
    dict(
        org="NucleusAI",
        name="NucleusX-1B-copyllama",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=8,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        use_retention=True,
        layernorm_embedding=False,
        use_lm_decay=False,
        no_scale_embedding=True,
        retnet_bottleneck=768,
    ),
    dict(
        org="NucleusAI",
        name="NucleusX-1B-copyllama-GQA",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=32,
        n_embd=2048,
        n_query_groups=4,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        use_retention=True,
        layernorm_embedding=False,
        use_lm_decay=False,
        no_scale_embedding=True,
    ),
    dict(
        org="syncdoth",
        name="RetNet-7B",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=16,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=6912,
        use_retention=True,
        layernorm_embedding=True,
        use_lm_decay=False,
        no_scale_embedding=False,
    ),
    dict(
        org="NucleusAI",
        name="NucleusX-7B",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=16,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        use_retention=True,
        retnet_bottleneck=1024,
        use_chunkwise_retention=True,
        layernorm_embedding=False,
        use_lm_decay=False,
        no_scale_embedding=True,
    ),
    dict(
        org="NucleusAI",
        name="MistRet-7B",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=16,
        n_embd=4096,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=14336,
        use_retention=True,
        retnet_bottleneck=1024,
        use_chunkwise_retention=True,
        layernorm_embedding=False,
        use_lm_decay=False,
        no_scale_embedding=True,
    ),
    dict(
        org="NucleusAI",
        name="RetNet-410m",  # NOTE: actually 430m
        block_size=2048,
        n_layer=24,
        n_embd=1024,
        n_head=4,
        padding_multiple=128,
        use_retention=True,
        layernorm_embedding=False,
        use_lm_decay=False,
        no_scale_embedding=True,
    ),
    dict(
        org="NucleusAI",
        name="RetNet-410m-Hybrid",  # NOTE: actually 430m
        block_size=2048,
        n_layer=24,
        n_embd=1024,
        n_head=4,
        padding_multiple=128,
        use_retention=True,
        layernorm_embedding=False,
        use_lm_decay=False,
        no_scale_embedding=True,
        hybrid_attention_layers=[1, 11],
        hybrid_att_window=None,  # TODO: change later: maybe 64 from based
    ),
    dict(org="NucleusAI",
         name="RetNeoX-1B",
         block_size=2048,
         n_layer=16,
         n_embd=2048,
         n_head=8,
         padding_multiple=128,
         use_retention=True,
         layernorm_embedding=False,
         use_lm_decay=False,
         no_scale_embedding=True,
    ),
    dict(org="NucleusAI",
         name="RetNeoX-1B-Hybrid",
         block_size=2048,
         n_layer=16,
         n_embd=2048,
         n_head=8,
         padding_multiple=128,
         use_retention=True,
         layernorm_embedding=False,
         use_lm_decay=False,
         no_scale_embedding=True,
         hybrid_attention_layers=[1, 7],
         hybrid_att_window=None,  # TODO: change later: maybe 64 from based
    ),
    dict(
        org="NucleusAI",
        name="RetNet-1_5",  # NOTE: actually 1.4B
        vocab_size=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_embd=2048,
        n_layer=24,
        n_head=8,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        retnet_bottleneck=768,
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
        use_retention=True,
        layernorm_embedding=False,
        use_lm_decay=False,
        no_scale_embedding=True,
    ),
    dict(
        org="NucleusAI",
        name="StripedMamba-410m-expand2",
        block_size=2048,
        n_layer=24,
        n_embd=1024,
        padding_multiple=128,
        use_mamba=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    ),
    dict(
        org="NucleusAI",
        name="StripedMamba-410m-Hybrid",
        block_size=2048,
        n_layer=24,
        n_embd=1024,
        padding_multiple=128,
        use_mamba=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,  # originally it should be 2. But to match the parameter size, we use 1.5
        hybrid_attention_layers=[1, 11],
        hybrid_att_window=None,  # TODO: change later: maybe 64 from based
    ),
]
configs.extend(retnet)

name_to_config = {config["name"]: config for config in configs}
