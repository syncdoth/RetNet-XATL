import os
import glob
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import fire


def get_weight_norms(base_path, mode, interval):
    all_ckpts = sorted(glob.glob(os.path.join(base_path, "*.pth")))
    start = int(os.path.basename(all_ckpts[0]).split("-")[1])
    end = int(os.path.basename(all_ckpts[-1]).split("-")[1])

    # FREEZE params
    all_modes = mode.split("-")
    freeze_weights = ["wte", "ln_f", "lm_head"]
    if "skip_reten" not in all_modes:
        freeze_weights.append("attn.reten")
    if "skip_mlp" not in all_modes:
        freeze_weights.extend(["mlp", "norm_2"])
    if "skip_oproj" not in all_modes:
        freeze_weights.extend(["attn.proj", "norm_1"])

    all_weight_norms = {}
    all_copied_weight_norms = {}
    for i in tqdm(range(start, end + interval, interval)):
        copied_weight_norms = []
        weight_norms = []
        path = os.path.join(base_path, f"iter-{i:06}-ckpt.pth")
        ckpt = torch.load(path)["model"]

        for key, param in ckpt.items():
            if any(freeze in key for freeze in freeze_weights):
                copied_weight_norms.append(torch.norm(param))
            else:
                weight_norms.append(torch.norm(param))

        all_weight_norms[i] = torch.mean(torch.tensor(weight_norms)).item()
        all_copied_weight_norms[i] = torch.mean(torch.tensor(copied_weight_norms)).item()
        del ckpt

    return all_weight_norms, all_copied_weight_norms


def main(
    copy_mode="skip_reten",
    model="RetNet-410m",
    freeze=True,
    base_path="checkpoints/{}-bs1024-pile_dedup-copy_exp-{}",
    interval=5000,
):
    base_path = base_path.format(model, copy_mode)
    if freeze:
        base_path += "-freeze"
    all_weight_norms, all_copied_weight_norms = get_weight_norms(base_path, copy_mode, interval)

    # find argmax
    if freeze:
        max_weight_norm = max(all_weight_norms, key=all_weight_norms.get)
        print(f"Max Weight Norm: step={max_weight_norm}, norm={all_weight_norms[max_weight_norm]}")

    # plot
    plt.plot(list(all_copied_weight_norms.keys()),
             list(all_copied_weight_norms.values()),
             label="Copied")
    plt.plot(list(all_weight_norms.keys()), list(all_weight_norms.values()), label="New")
    if freeze:
        plt.axvline(x=max_weight_norm, color="r", linestyle="--", label="Max Weight Norm")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Weight Norm")
    plt.title(f"{model}-{copy_mode}{'-freeze' if freeze else ''} Weight Norms")

    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{model}-{copy_mode}{'-freeze' if freeze else ''}-weight_norms.png")


if __name__ == "__main__":
    fire.Fire(main)
