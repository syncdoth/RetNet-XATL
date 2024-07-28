import os
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import fire
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from convert_lit_checkpoint import convert_lit_checkpoint
from convert_to_safetensors import main as convert_to_safetensors
from hf_retnet.configuration_retnet import RetNetConfig
from hf_retnet.modeling_retnet import RetNetModel, RetNetForCausalLM

AutoConfig.register("retnet", RetNetConfig)
AutoModel.register(RetNetConfig, RetNetModel)
AutoModelForCausalLM.register(RetNetConfig, RetNetForCausalLM)

RetNetConfig.register_for_auto_class()
RetNetModel.register_for_auto_class("AutoModel")
RetNetForCausalLM.register_for_auto_class("AutoModelForCausalLM")


def main(checkpoint_name: str, out_dir: Path, model_name: str, do_upload: bool = True):
    convert_lit_checkpoint(checkpoint_name=checkpoint_name,
                           out_dir=Path(out_dir),
                           model_name=model_name,
                           model_only=False)
    name = os.path.splitext(checkpoint_name)[0]
    ckpt_path = os.path.join(out_dir, name + ".bin")
    upload_name = os.path.basename(out_dir)

    os.makedirs(os.path.join(out_dir, f"hf-{name}"), exist_ok=True)
    os.rename(ckpt_path, os.path.join(out_dir, f"hf-{name}", "pytorch_model.bin"))
    os.rename(os.path.join(out_dir, "config.json"),
              os.path.join(out_dir, f"hf-{name}", "config.json"))

    convert_to_safetensors(os.path.join(out_dir, f"hf-{name}"), os.path.join(out_dir, f"hf-{name}"), delete_old=True)

    if do_upload:
        if "RetNet" in model_name:
            model = RetNetForCausalLM.from_pretrained(os.path.join(out_dir, f"hf-{name}"))
        else:
            model = AutoModelForCausalLM.from_pretrained(os.path.join(out_dir, f"hf-{name}"))

        if "410m" in model_name:
            tokenizer_path = "EleutherAI/pythia-410m"
        else:
            raise NotImplementedError
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.push_to_hub(f"NucleusAI/{upload_name}", revision=name, private=True)
        model.push_to_hub(f"NucleusAI/{upload_name}", revision=name, private=True)


if __name__ == "__main__":
    fire.Fire(main)
