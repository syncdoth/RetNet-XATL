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


def convert_llama_to_retnet(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    skip_reten: Optional[bool] = False,
    skip_oproj: Optional[bool] = False,
    skip_mlp: Optional[bool] = False,
):
    weight_map = {
        "transformer.h.{}.attn.attn.weight": "transformer.h.{}.attn.reten.weight",
        "transformer.h.{}.attn.attn.bias": "transformer.h.{}.attn.reten.bias",
    }
    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map.get(from_name)

            if (config.hybrid_attention_layers is not None and
                    number in config.hybrid_attention_layers and "attn" in from_name):
                to_name = None
            else:
                if skip_reten and "attn.attn" in from_name:
                    continue
                if skip_oproj and "attn.proj" in from_name:
                    continue
                if skip_mlp and "mlp" in from_name:
                    continue

            if to_name is None:
                to_name = from_name
            to_name = to_name.format(number)
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param
        else:
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[name] = param


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


@torch.inference_mode()
def convert_lit_checkpoint(*,
                           checkpoint_name: str,
                           out_dir: Path,
                           model_name: str,
                           skip_reten: bool = False,
                           skip_oproj: bool = False,
                           skip_mlp: bool = False) -> None:
    config = Config.from_name(model_name)

    copy_fn = partial(convert_llama_to_retnet,
                      config,
                      skip_reten=skip_reten,
                      skip_oproj=skip_oproj,
                      skip_mlp=skip_mlp)

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # checkpoint_name cannot be hardcoded because there exists different outputs such as
    # ("lit_model_finetuned.pth", "lit_model_lora_finetuned.pth", "lit_model_adapter_finetuned.pth"")
    pth_file = out_dir / checkpoint_name
    save_dir = out_dir / "retnet"
    save_dir.mkdir(exist_ok=True, parents=True)

    save_name = "lit_model"
    if skip_reten:
        save_name += "-skip_reten"
    if skip_oproj:
        save_name += "-skip_oproj"
    if skip_mlp:
        save_name += "-skip_mlp"
    if config.hybrid_attention_layers is not None:
        save_name += f"-Hybrid"
    bin_file = save_dir / f"{save_name}.bin"

    with incremental_save(bin_file) as saver:
        with contextlib.ExitStack() as stack:
            lit_weights = stack.enter_context(lazy_load(pth_file))
            lit_weights = maybe_unwrap_state_dict(lit_weights)
            check_conversion_supported(lit_weights)
            # Incremental save will trigger error
            copy_fn(sd, lit_weights, saver=None)
            gc.collect()
        saver.save(sd)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint, as_positional=False)
