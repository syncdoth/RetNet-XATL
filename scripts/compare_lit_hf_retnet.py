import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import asdict

import fire
import torch
from transformers import AutoTokenizer

from hf_retnet.modeling_retnet import RetNetForCausalLM
from hf_retnet.configuration_retnet import RetNetConfig
from lit_gpt import GPT, Config

from convert_lit_checkpoint import (maybe_unwrap_state_dict, check_conversion_supported,
                                    copy_weights_retnet, convert_config_lit_to_retnet)

def load_hf_model_from_lit(lit_model, lit_config):
    hf_sd = {}
    lit_weights = lit_model.state_dict()
    lit_weights = maybe_unwrap_state_dict(lit_weights)
    check_conversion_supported(lit_weights)
    copy_weights_retnet(lit_config, hf_sd, lit_weights)

    hf_config_dict = convert_config_lit_to_retnet(asdict(lit_config))
    hf_config = RetNetConfig(**hf_config_dict)
    hf_model = RetNetForCausalLM(hf_config)
    load_status = hf_model.load_state_dict(hf_sd, strict=False)
    assert load_status.missing_keys == ['model.retnet_rel_pos.angle', 'model.retnet_rel_pos.decay'], load_status.missing_keys
    assert not load_status.unexpected_keys, load_status.unexpected_keys

    return hf_model

def main(
    test_text="Hello darkness my old friend,",
    checkpoint_pth=None,
    model_name="RetNet-410m",
    tokenizer="EleutherAI/pythia-410m",
    device="cuda",
    dtype=torch.float32,
):
    lit_config = Config.from_name(model_name)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    lit_model = GPT(lit_config, dtype=dtype)
    if checkpoint_pth is None:
        hf_model = load_hf_model_from_lit(lit_model, lit_config)
    else:
        lit_model.load_state_dict(torch.load(checkpoint_pth)["model"])
        fname = os.path.splitext(os.path.basename(checkpoint_pth))[0]
        hf_model_dir = os.path.join(os.path.dirname(checkpoint_pth), f"hf-{fname}")
        if not os.path.isdir(hf_model_dir):
            raise ValueError(f"huggingface model `{hf_model_dir}` not converted yet")
        hf_model = RetNetForCausalLM.from_pretrained(hf_model_dir, attn_implementation="sdpa")

    lit_model = lit_model.to(device, dtype=dtype)
    hf_model = hf_model.to(device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.add_bos_token = True  # used in training

    input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to(device)

    lit_logits = lit_model(input_ids)
    hf_logits = hf_model(input_ids).logits

    # print differences
    print("before softmax (logits)")
    print("lit", lit_logits)
    print("hf", hf_logits)

    print("after softmax (probs)")
    lit_prob = torch.softmax(lit_logits, dim=-1)
    hf_prob = torch.softmax(hf_logits, dim=-1)
    print("lit", lit_prob)
    print("hf", hf_prob)

    print("prob differences")
    prob_diff = (lit_prob - hf_prob).abs()
    print("mean:", prob_diff.mean().item(), "max:", prob_diff.max().item(), "var:", prob_diff.var().item())

if __name__ == "__main__":
    fire.Fire(main)
