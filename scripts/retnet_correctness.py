import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import fire
import torch
from transformers import AutoTokenizer

from hf_retnet.modeling_retnet import RetNetForCausalLM


def main(
    test_text="Hello darkness my old friend,",
    checkpoint_pth=None,
    tokenizer="EleutherAI/pythia-410m",
    device="cuda",
    dtype=torch.float32,
    test_logits=True,
    test_past_kv=True,
    test_generation=True,
):
    hf_model = RetNetForCausalLM.from_pretrained(checkpoint_pth)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    hf_model = hf_model.to(device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.add_bos_token = True  # used in training

    input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to(device)

    # forwards
    parallel_output = hf_model(input_ids, forward_impl="parallel", use_cache=True)
    parallel_logits = parallel_output.logits
    parallel_past_kv = parallel_output.past_key_values

    recurrent_logits = []
    past_key_values = None
    for i in range(input_ids.shape[1]):
        token = input_ids[:, :i+1]
        outputs = hf_model(token, use_cache=True, forward_impl="recurrent", past_key_values=past_key_values)
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        recurrent_logits.append(logits)
    recurrent_logits = torch.cat(recurrent_logits, dim=1)

    chunk_output = hf_model(input_ids, forward_impl="chunkwise", use_cache=True, recurrent_chunk_size=2)
    chunk_logits = chunk_output.logits
    chunk_past_kv = chunk_output.past_key_values

    if test_logits:
        par_vs_rnn = torch.allclose(parallel_logits, recurrent_logits, atol=1e-5)
        par_vs_chunk = torch.allclose(parallel_logits, chunk_logits, atol=1e-5)
        rnn_vs_chunk = torch.allclose(recurrent_logits, chunk_logits, atol=1e-5)

        if par_vs_rnn and par_vs_chunk and rnn_vs_chunk:
            print("all logits same!")
        else:
            print(
                f"par_vs_rnn: {par_vs_rnn}\n"
                f"par_vs_chunk: {par_vs_chunk}\n"
                f"rnn_vs_chunk: {rnn_vs_chunk}\n"
            )
            print("parallel\n", parallel_logits)
            print("recurrent\n", recurrent_logits)
            print("chunk\n", chunk_logits)

    if test_past_kv:
        parallel_past_kv = torch.stack([x["prev_key_value"] for x in parallel_past_kv], 0)
        recurrent_past_kv = torch.stack([x["prev_key_value"] for x in past_key_values], 0)
        chunk_past_kv = torch.stack([x["prev_key_value"] for x in chunk_past_kv], 0)

        par_vs_rnn = torch.allclose(parallel_past_kv, recurrent_past_kv, atol=1e-5)
        par_vs_chunk = torch.allclose(parallel_past_kv, chunk_past_kv, atol=1e-5)
        rnn_vs_chunk = torch.allclose(recurrent_past_kv, chunk_past_kv, atol=1e-5)

        if par_vs_rnn and par_vs_chunk and rnn_vs_chunk:
            print("all past_kv same!")
        else:
            print(
                f"par_vs_rnn: {par_vs_rnn}\n"
                f"par_vs_chunk: {par_vs_chunk}\n"
                f"rnn_vs_chunk: {rnn_vs_chunk}\n"
            )
            print("parallel_past_kv\n", parallel_past_kv[0])
            print("rnn_past_kv\n", recurrent_past_kv[0])
            print("chunk_past_kv\n", chunk_past_kv[0])

    if test_generation:
        outputs_par = hf_model.generate(input_ids, max_new_tokens=20, use_cache=False)
        outputs_rec = hf_model.generate(input_ids, max_new_tokens=20, use_cache=True)
        print("parallel\n", tokenizer.decode(outputs_par[0]))
        print("recurrent\n", tokenizer.decode(outputs_rec[0]))

if __name__ == "__main__":
    fire.Fire(main)
