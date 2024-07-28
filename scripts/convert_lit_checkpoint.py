import contextlib
import gc
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union
from dataclasses import asdict
import json
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.utils import NotYetLoadedTensor, incremental_save, lazy_load
# from scripts.convert_hf_checkpoint import layer_template, load_param


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(param: Union[torch.Tensor, NotYetLoadedTensor], name: str,
               dtype: Optional[torch.dtype]) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
        print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param


def copy_weights_falcon(
    config,
    size: Literal["7b", "40b"],
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "transformer.wte.weight":
            "transformer.word_embeddings.weight",
        "transformer.h.{}.attn.attn.weight":
            "transformer.h.{}.self_attention.query_key_value.weight",
        "transformer.h.{}.attn.proj.weight":
            "transformer.h.{}.self_attention.dense.weight",
        "transformer.h.{}.mlp.mlp.fc1.weight":
            "transformer.h.{}.mlp.dense_h_to_4h.weight",
        "transformer.h.{}.mlp.mlp.fc2.weight":
            "transformer.h.{}.mlp.dense_4h_to_h.weight",
        "transformer.ln_f.bias":
            "transformer.ln_f.bias",
        "transformer.ln_f.weight":
            "transformer.ln_f.weight",
        "lm_head.weight":
            "lm_head.weight",
    }
    # the original model definition is different for each size
    if size == "7b":
        weight_map.update({
            "transformer.h.{}.norm_1.bias": "transformer.h.{}.input_layernorm.bias",
            "transformer.h.{}.norm_1.weight": "transformer.h.{}.input_layernorm.weight",
        })
    elif size == "40b":
        weight_map.update({
            "transformer.h.{}.norm_1.bias": "transformer.h.{}.ln_attn.bias",
            "transformer.h.{}.norm_1.weight": "transformer.h.{}.ln_attn.weight",
            "transformer.h.{}.norm_2.bias": "transformer.h.{}.ln_mlp.bias",
            "transformer.h.{}.norm_2.weight": "transformer.h.{}.ln_mlp.weight",
        })
    else:
        raise NotImplementedError

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        # if name in ("transformer.wte.weight", "lm_head.weight"):
        #     if param.shape[0] > config.vocab_size:
        #         print(f"Trimming {name!r} from {param.shape[0]} to {config.vocab_size} vocabs")
        #         param = param[:config.vocab_size, :]
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_striped_mamba(
    config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "gpt_neox.embed_in.weight",
        "transformer.h.{}.norm_1.bias": "gpt_neox.layers.{}.input_layernorm.bias",
        "transformer.h.{}.norm_1.weight": "gpt_neox.layers.{}.input_layernorm.weight",
        # Mamba Block
        "transformer.h.{}.attn.in_proj.bias": "gpt_neox.layers.{}.attention.mamba.in_proj.bias",
        "transformer.h.{}.attn.in_proj.weight": "gpt_neox.layers.{}.attention.mamba.in_proj.weight",
        "transformer.h.{}.attn.conv1d.bias": "gpt_neox.layers.{}.attention.mamba.conv1d.bias",
        "transformer.h.{}.attn.conv1d.weight": "gpt_neox.layers.{}.attention.mamba.conv1d.weight",
        "transformer.h.{}.attn.x_proj.bias": "gpt_neox.layers.{}.attention.mamba.x_proj.bias",
        "transformer.h.{}.attn.x_proj.weight": "gpt_neox.layers.{}.attention.mamba.x_proj.weight",
        "transformer.h.{}.attn.dt_proj.bias": "gpt_neox.layers.{}.attention.mamba.dt_proj.bias",
        "transformer.h.{}.attn.dt_proj.weight": "gpt_neox.layers.{}.attention.mamba.dt_proj.weight",
        "transformer.h.{}.attn.out_proj.bias": "gpt_neox.layers.{}.attention.mamba.out_proj.bias",
        "transformer.h.{}.attn.out_proj.weight": "gpt_neox.layers.{}.attention.mamba.out_proj.weight",
        "transformer.h.{}.attn.A_log": "gpt_neox.layers.{}.attention.mamba.A_log",
        "transformer.h.{}.attn.D": "gpt_neox.layers.{}.attention.mamba.D",
        #################
        "transformer.h.{}.norm_2.bias": "gpt_neox.layers.{}.post_attention_layernorm.bias",
        "transformer.h.{}.norm_2.weight": "gpt_neox.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.mlp.fc1.bias": "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias",
        "transformer.h.{}.mlp.mlp.fc1.weight": "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight",
        "transformer.h.{}.mlp.mlp.fc2.bias": "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias",
        "transformer.h.{}.mlp.mlp.fc2.weight": "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight",
        "transformer.ln_f.bias": "gpt_neox.final_layer_norm.bias",
        "transformer.ln_f.weight": "gpt_neox.final_layer_norm.weight",
        "lm_head.weight": "embed_out.weight",
    }

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            if "mlp.fc." in from_name:
                from_name = from_name.replace("mlp.fc.", "mlp.mlp.fc1.")
            elif "mlp.proj." in from_name:
                from_name = from_name.replace("mlp.proj.", "mlp.mlp.fc2.")
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        # if name in ("transformer.wte.weight", "lm_head.weight"):
        #     if param.shape[0] > config.vocab_size:
        #         print(f"Trimming {name!r} from {param.shape[0]} to {config.vocab_size} vocabs")
        #         param = param[:config.vocab_size, :]
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_gpt_neox(
    config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "gpt_neox.embed_in.weight",
        "transformer.h.{}.norm_1.bias": "gpt_neox.layers.{}.input_layernorm.bias",
        "transformer.h.{}.norm_1.weight": "gpt_neox.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.attn.bias": "gpt_neox.layers.{}.attention.query_key_value.bias",
        "transformer.h.{}.attn.attn.weight": "gpt_neox.layers.{}.attention.query_key_value.weight",
        "transformer.h.{}.attn.proj.bias": "gpt_neox.layers.{}.attention.dense.bias",
        "transformer.h.{}.attn.proj.weight": "gpt_neox.layers.{}.attention.dense.weight",
        "transformer.h.{}.norm_2.bias": "gpt_neox.layers.{}.post_attention_layernorm.bias",
        "transformer.h.{}.norm_2.weight": "gpt_neox.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.mlp.fc1.bias": "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias",
        "transformer.h.{}.mlp.mlp.fc1.weight": "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight",
        "transformer.h.{}.mlp.mlp.fc2.bias": "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias",
        "transformer.h.{}.mlp.mlp.fc2.weight": "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight",
        "transformer.ln_f.bias": "gpt_neox.final_layer_norm.bias",
        "transformer.ln_f.weight": "gpt_neox.final_layer_norm.weight",
        "lm_head.weight": "embed_out.weight",
    }

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            if "mlp.fc." in from_name:
                from_name = from_name.replace("mlp.fc.", "mlp.mlp.fc1.")
            elif "mlp.proj." in from_name:
                from_name = from_name.replace("mlp.proj.", "mlp.mlp.fc2.")
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        # if name in ("transformer.wte.weight", "lm_head.weight"):
        #     if param.shape[0] > config.vocab_size:
        #         print(f"Trimming {name!r} from {param.shape[0]} to {config.vocab_size} vocabs")
        #         param = param[:config.vocab_size, :]
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_llama(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "transformer.h.{}.norm_1.weight": "model.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.proj.weight": "model.layers.{}.self_attn.o_proj.weight",
        "transformer.h.{}.norm_2.weight": "model.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.swiglu.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
        "transformer.h.{}.mlp.swiglu.w2.weight": "model.layers.{}.mlp.up_proj.weight",
        "transformer.h.{}.mlp.swiglu.w3.weight": "model.layers.{}.mlp.down_proj.weight",
        "transformer.ln_f.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for name, param in lit_weights.items():
        if name.endswith(".attn.attn.weight"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.self_attn.q_proj.weight".format(number)
            k = "model.layers.{}.self_attn.k_proj.weight".format(number)
            v = "model.layers.{}.self_attn.v_proj.weight".format(number)
            qkv = load_param(param, name, None)
            qp, kp, vp = tensor_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        elif "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]

            if to_name is None:
                continue
            to_name = to_name.format(number)
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param

        else:
            to_name = weight_map[name]
            param = load_param(param, name, None)
            # if name in ("transformer.wte.weight", "lm_head.weight"):
            #     if param.shape[0] > config.vocab_size:
            #         print(f"Trimming {name!r} from {param.shape[0]} to {config.vocab_size} vocabs")
            #         param = param[:config.vocab_size, :]
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def copy_weights_retnet(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    if config._mlp_class == "LLaMAMLP":
        mlp_weight_map = {
            # NOTE: in torchscale's implementation, the gate and up_proj is swapped w.r.t
            # swiglu in LlamaMLP. Hence, the mapping is different.
            # in llama:
            #     "transformer.h.{}.mlp.swiglu.w1.weight": "model.layers.{}.ffn.gate.weight",
            #     "transformer.h.{}.mlp.swiglu.w2.weight": "model.layers.{}.ffn.fc1.weight",
            "transformer.h.{}.mlp.swiglu.w1.weight": "model.layers.{}.ffn.fc1.weight",
            "transformer.h.{}.mlp.swiglu.w2.weight": "model.layers.{}.ffn.gate.weight",
            "transformer.h.{}.mlp.swiglu.w3.weight": "model.layers.{}.ffn.fc2.weight",
        }
    else:
        mlp_weight_map = {
            "transformer.h.{}.mlp.mlp.fc1.weight": "model.layers.{}.ffn.fc1.weight",
            "transformer.h.{}.mlp.mlp.fc2.weight": "model.layers.{}.ffn.fc2.weight",
        }

    if config.bias:
        bias_map = {
            "ln_e.bias": "model.layernorm_embedding.bias",
            "transformer.h.{}.norm_1.bias": "model.layers.{}.retention_layer_norm.bias",
            "transformer.h.{}.norm_2.bias": "model.layers.{}.final_layer_norm.bias",
            "transformer.h.{}.attn.group_norm.bias": "model.layers.{}.retention.group_norm.bias",
            "transformer.h.{}.attn.proj.bias": "model.layers.{}.retention.out_proj.bias",
            "transformer.h.{}.attn.gate_proj.bias": "model.layers.{}.retention.g_proj.bias",
            "transformer.ln_f.bias": "model.layer_norm.bias",
        }
        if config._mlp_class == "LLaMAMLP":
            bias_map.update({
                "transformer.h.{}.mlp.swiglu.w1.bias": "model.layers.{}.ffn.fc1.bias",
                "transformer.h.{}.mlp.swiglu.w2.bias": "model.layers.{}.ffn.gate.bias",
                "transformer.h.{}.mlp.swiglu.w3.bias": "model.layers.{}.ffn.fc2.bias",
            })
        else:
            bias_map.update({
                "transformer.h.{}.mlp.mlp.fc1.bias": "model.layers.{}.ffn.fc1.bias",
                "transformer.h.{}.mlp.mlp.fc2.bias": "model.layers.{}.ffn.fc2.bias",
            })
    else:
        bias_map = {}

    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "ln_e.weight": "model.layernorm_embedding.weight",
        "transformer.h.{}.norm_1.weight": "model.layers.{}.retention_layer_norm.weight",
        "transformer.h.{}.norm_2.weight": "model.layers.{}.final_layer_norm.weight",
        "transformer.h.{}.attn.group_norm.weight": "model.layers.{}.retention.group_norm.weight",
        "transformer.h.{}.attn.proj.weight": "model.layers.{}.retention.out_proj.weight",
        "transformer.h.{}.attn.gate_proj.weight": "model.layers.{}.retention.g_proj.weight",
        "transformer.h.{}.attn.down_proj.weight": "model.layers.{}.retention.down_proj.weight",
        "transformer.ln_f.weight": "model.layer_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    weight_map.update(mlp_weight_map)
    weight_map.update(bias_map)

    for name, param in lit_weights.items():
        if name.endswith(".attn.reten.weight"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.retention.q_proj.weight".format(number)
            k = "model.layers.{}.retention.k_proj.weight".format(number)
            v = "model.layers.{}.retention.v_proj.weight".format(number)
            qkv = load_param(param, name, None)
            qp, kp, vp = tensor_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        elif name.endswith(".attn.reten.bias"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.retention.q_proj.bias".format(number)
            k = "model.layers.{}.retention.k_proj.bias".format(number)
            v = "model.layers.{}.retention.v_proj.bias".format(number)
            qkv = load_param(param, name, None)
            qkv = qkv.unsqueeze(-1)
            qp, kp, vp = [x.squeeze() for x in tensor_split(qkv, config)]
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param

        elif name.endswith(".attn.attn.weight"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.retention.q_proj.weight".format(number)
            k = "model.layers.{}.retention.k_proj.weight".format(number)
            v = "model.layers.{}.retention.v_proj.weight".format(number)
            qkv = load_param(param, name, None)
            qp, kp, vp = tensor_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        elif name.endswith(".attn.attn.bias"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.retention.q_proj.bias".format(number)
            k = "model.layers.{}.retention.k_proj.bias".format(number)
            v = "model.layers.{}.retention.v_proj.bias".format(number)
            qkv = load_param(param, name, None)
            qkv = qkv.unsqueeze(-1)
            qp, kp, vp = [x.squeeze() for x in tensor_split(qkv, config)]
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param


        elif "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]

            if to_name is None:
                continue
            to_name = to_name.format(number)
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param

        else:
            to_name = weight_map[name]
            param = load_param(param, name, None)
            # if name in ("transformer.wte.weight", "lm_head.weight"):
            #     if param.shape[0] > config.vocab_size:
            #         print(f"Trimming {name!r} from {param.shape[0]} to {config.vocab_size} vocabs")
            #         param = param[:config.vocab_size, :]
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def tensor_split(param: Union[torch.Tensor, NotYetLoadedTensor],
                 config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    def kstart(start, blen, klen) -> int:
        """returns start index of keys in batch"""
        return start + (blen - (klen * 2))

    def vstart(start, blen, klen) -> int:
        """returns start index of values in batch"""
        return start + blen - klen

    def vend(start, blen) -> int:
        """returns last index of values in batch"""
        return start + blen

    # num observations
    nobs = param.shape[0]
    # batch length
    blen = nobs // config.n_query_groups
    # key length in batch
    klen = config.head_size
    # value length in batch
    vlen = config.head_size
    # the starting index of each new batch
    starts = range(0, nobs, blen)
    # the indices to splice on
    splices = [(s, kstart(s, blen, klen), vstart(s, blen, vlen), vend(s, blen)) for s in starts]

    qc = ()
    kc = ()
    vc = ()

    for splice in splices:
        qs, ks, vs, ve = splice
        qc += (param[qs:ks, :],)
        kc += (param[ks:vs, :],)
        vc += (param[vs:ve, :],)

    q = torch.cat(qc)
    k = torch.cat(kc)
    v = torch.cat(vc)

    return q, k, v


def maybe_unwrap_state_dict(lit_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return lit_weights.get("model", lit_weights)


def check_conversion_supported(lit_weights: Dict[str, torch.Tensor]) -> None:
    weight_names = {wk.split(".")[-1] for wk in lit_weights}
    # LoRA or QLoRA
    if any("lora" in wn for wn in weight_names):
        raise ValueError(
            "Model weights must be merged using `lora.merge_lora_weights()` before conversion.")
    # adapter v2. adapter_bias will only be in adapter_v2
    elif "adapter_bias" in weight_names:
        raise NotImplementedError("Converting models finetuned with adapter_v2 not yet supported.")
    # adapter. gating_factor is in adapter and adapter_v2
    elif "gating_factor" in weight_names:
        raise NotImplementedError("Converting models finetuned with adapter not yet supported.")


def get_tinyllama_init_hf_config() -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": None,
        "initializer_range": 0.02,
        "intermediate_size": None,
        "max_position_embeddings": None,
        "model_type": "llama",
        "num_attention_heads": None,
        "num_hidden_layers": None,
        "num_key_value_heads": None,
        "pretraining_tp": 1,
        "rms_norm_eps": None,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.31.0.dev0",
        "use_cache": True,
        "vocab_size": None,
    }


def get_gpt_neox_init_hf_config() -> dict:
    return {
        "architectures": ["GPTNeoXForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 0,
        "hidden_act": "gelu",
        "hidden_size": None,
        "initializer_range": 0.02,
        "intermediate_size": None,
        "layer_norm_eps": None,
        "max_position_embeddings": None,
        "model_type": "gpt_neox",
        "num_attention_heads": None,
        "num_hidden_layers": None,
        "rotary_emb_base": 10000,
        "rotary_pct": 0.25,
        "tie_word_embeddings": False,
        "transformers_version": "4.34.1",
        "use_cache": True,
        "use_parallel_residual": True,
        "vocab_size": None
    }

def get_striped_mamba_init_hf_config() -> dict:
    return {
        "architectures": ["StripedMambaForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 0,
        "hidden_act": "gelu",
        "hidden_size": None,
        "initializer_range": 0.02,
        "intermediate_size": None,
        "layer_norm_eps": None,
        "max_position_embeddings": None,
        "model_type": "striped_mamba",
        "num_attention_heads": None,
        "num_hidden_layers": None,
        "rotary_emb_base": 10000,
        "rotary_pct": 0.25,
        "tie_word_embeddings": False,
        "transformers_version": "4.34.1",
        "use_cache": True,
        "use_parallel_residual": True,
        "vocab_size": None
    }


def get_phi_init_hf_config() -> dict:
    return {
        "_name_or_path": "microsoft/phi-1_5",
        "architectures": ["PhiForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": None,
        "embd_pdrop": 0.0,
        "eos_token_id": None,
        "hidden_act": "gelu_new",
        "hidden_size": None,
        "initializer_range": 0.02,
        "intermediate_size": None,
        "layer_norm_eps": None,
        "max_position_embeddings": None,
        "model_type": "phi",
        "num_attention_heads": None,
        "num_hidden_layers": None,
        "num_key_value_heads": None,
        "partial_rotary_factor": 0.5,
        "qk_layernorm": None,
        "resid_pdrop": 0.0,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "transformers_version": "4.37.0",
        "use_cache": True,
        "vocab_size": None
    }


def get_retnet_init_hf_config() -> dict:
    return {
        "architectures": ["RetNetForCausalLM"],
        "model_type": "retnet",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": None,
        "activation_dropout": 0.0,
        "drop_path_rate": 0.0,
        "dropout": 0.0,
        "deepnorm": False,
        "forward_impl": "parallel",
        "decoder_normalize_before": True,
        "activation_fn": "swish",
        "is_decoder": True,
        "output_retentions": False,
        "recurrent_chunk_size": 512,
        "subln": False,
        "tie_word_embeddings": False,
        "initializer_range": 0.02,
        "transformers_version": "4.34.1",
        "use_cache": False,
        "use_glu": True,
        "z_loss_coeff": 0.0,
        "use_ffn_rms_norm": False,
        "rotary_percentage": 1.0,
        "parallel_residual": False,
        "use_bias": False,
        "use_rms_norm": True,
        "groupnorm_affine": False,
        "shared_attention_norm": False,
        "lm_head_bias": False,
    }


def convert_config_lit_to_hf(lit_config_dict: dict, config: Config) -> dict:
    lit_hf_mapping = {
        "block_size": "max_position_embeddings",
        "padded_vocab_size": "vocab_size",
        "n_layer": "num_hidden_layers",
        "n_embd": "hidden_size",
        "n_head": "num_attention_heads",
        "intermediate_size": "intermediate_size",
    }
    if config._mlp_class == "LLaMAMLP":
        lit_hf_mapping.update({
            "n_query_groups": "num_key_value_heads",
            "norm_eps": "rms_norm_eps",
        })
        hf_config_dict = get_tinyllama_init_hf_config()
    else:
        lit_hf_mapping.update({
            "norm_eps": "layer_norm_eps",
        })
        if "phi" in config.name:
            hf_config_dict = get_phi_init_hf_config()
        if "Mamba" in config.name:
            hf_config_dict = get_striped_mamba_init_hf_config()
        else:
            hf_config_dict = get_gpt_neox_init_hf_config()

    for lit_key, hf_key in lit_hf_mapping.items():
        hf_config_dict[hf_key] = lit_config_dict[lit_key]
    return hf_config_dict


def convert_config_lit_to_retnet(lit_config_dict: dict) -> dict:
    lit_hf_mapping = {
        "block_size": "max_position_embeddings",
        "padded_vocab_size": "vocab_size",
        "n_layer": "decoder_layers",
        "n_embd": "decoder_embed_dim",
        "n_head": "decoder_retention_heads",
        # "n_query_groups": "num_key_value_heads",  # TODO: add later
        "intermediate_size": "decoder_ffn_embed_dim",
        "norm_eps": "layernorm_eps",
        "layernorm_embedding": "layernorm_embedding",
        "no_scale_embedding": "no_scale_embedding",
        "use_lm_decay": "use_lm_decay",
        "parallel_residual": "parallel_residual",  # TODO: add this in the retnet modeling code
        "bias": "use_bias",
        "rotary_percentage": "rotary_percentage",
        "shared_attention_norm": "shared_attention_norm",
        "lm_head_bias": "lm_head_bias",
        "retnet_bottleneck": "retnet_bottleneck",
        "use_ln_for_groupnorm": "groupnorm_affine",
        "hybrid_attention_layers": "hybrid_attention_layers",
    }
    hf_config_dict = get_retnet_init_hf_config()

    for lit_key, hf_key in lit_hf_mapping.items():
        hf_config_dict[hf_key] = lit_config_dict[lit_key]
    hf_config_dict["decoder_value_embed_dim"] = hf_config_dict["decoder_embed_dim"]
    hf_config_dict["use_glu"] = lit_config_dict["_mlp_class"] == "LLaMAMLP"
    hf_config_dict[
        "activation_fn"] = "swish" if lit_config_dict["_mlp_class"] == "LLaMAMLP" else "gelu_new"
    hf_config_dict["use_rms_norm"] = "RMSNorm" in lit_config_dict["_norm_class"]
    return hf_config_dict


@torch.inference_mode()
def convert_lit_checkpoint(*,
                           checkpoint_name: str,
                           out_dir: Path,
                           model_name: str,
                           model_only: bool = True) -> None:
    config = Config.from_name(model_name)

    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, config, "40b" if config.n_embd == 8192 else "7b")
    elif "RetNet" in model_name or "NucleusX" in model_name or "RetNeoX" in model_name:
        copy_fn = partial(copy_weights_retnet, config)
    elif config._mlp_class == "LLaMAMLP":
        copy_fn = partial(copy_weights_llama, config)
    elif "Mamba" in model_name:
        copy_fn = partial(copy_weights_striped_mamba, config)
    else:
        copy_fn = partial(copy_weights_gpt_neox, config)

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # checkpoint_name cannot be hardcoded because there exists different outputs such as
    # ("lit_model_finetuned.pth", "lit_model_lora_finetuned.pth", "lit_model_adapter_finetuned.pth"")
    pth_file = out_dir / checkpoint_name
    bin_file = pth_file.with_suffix(".bin")

    with incremental_save(bin_file) as saver:
        with contextlib.ExitStack() as stack:
            lit_weights = stack.enter_context(lazy_load(pth_file))
            lit_weights = maybe_unwrap_state_dict(lit_weights)
            check_conversion_supported(lit_weights)
            # Incremental save will trigger error
            copy_fn(sd, lit_weights, saver=None)
            gc.collect()
        saver.save(sd)

    # convert lit config file to hf-style
    if not model_only:
        print('Converting config file...')
        lit_config = asdict(config)
        if "RetNet" in model_name or "NucleusX" in model_name or "RetNeoX" in model_name:
            hf_config = convert_config_lit_to_retnet(lit_config)
        else:
            hf_config = convert_config_lit_to_hf(lit_config, config)
        config_path = out_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(hf_config, f, indent=4)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint, as_positional=False)
