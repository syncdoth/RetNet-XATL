"""
Adapted from https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch

import transformers
from transformers import AutoTokenizer

from hf_retnet.modeling_retnet import RetNetForCausalLM
from hf_retnet.modeling_smamba import StripedMambaForCausalLM

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("retnet")
class RetNetEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self,
                 pretrained="NucleusAI/RetNet-410m-bs1024-pile_dedup",
                 tokenizer="EleutherAI/pythia-410m",
                 max_length=2048,
                 batch_size=None,
                 device="cuda",
                 dtype=torch.float32):
        LM.__init__(self)
        self._model = RetNetForCausalLM.from_pretrained(pretrained,
                                                        use_cache=False,
                                                        attn_implementation="sdpa",
                                                        torch_dtype=dtype).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_bos_token = True
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


@register_model("striped_mamba")
class StripedMambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self,
                 pretrained="NucleusAI/StripedMamba-410m-bs1024-pile_dedup",
                 tokenizer="EleutherAI/pythia-410m",
                 max_length=2048,
                 batch_size=None,
                 device="cuda",
                 dtype=torch.float32):
        LM.__init__(self)
        self._model = StripedMambaForCausalLM.from_pretrained(pretrained,
                                                              torch_dtype=dtype).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_bos_token = True
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()

if __name__ == "__main__":
    cli_evaluate()
