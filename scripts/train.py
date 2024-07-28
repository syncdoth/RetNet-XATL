import sys
from pathlib import Path

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import glob
import math
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import lightning as L
import torch
from jsonargparse import CLI
from lightning.fabric.strategies import DDPStrategy, FSDPStrategy, XLAStrategy
from lion_pytorch import Lion
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lit_gpt import FusedCrossEntropyLoss

from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import (chunked_cross_entropy, get_default_supported_precision, num_parameters,
                           step_csv_logger)
from custom_datasets.refined_web import RefinedWebDataset
from custom_datasets.slimpajama import SlimPajamaDataset
from custom_datasets.star_coder import StarCoderDataset
from custom_datasets.pile_deduplicated import PileDedupDataset

data_class_map = {
    "refinedweb": RefinedWebDataset,
    "train_slim": SlimPajamaDataset,
    "train_star": StarCoderDataset,
    "pile_dedup": PileDedupDataset,
}

TORCH_MATMUL_PRECISION = "high"


@dataclass
class HyperParameters:
    # Hyperparameters
    ########### Environment ###########
    num_of_devices: int = 8
    num_of_nodes: int = 4

    ########### optimizer ###########
    optimizer_choice: str = "adamw"
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    learning_rate: float = 4e-4
    decay_lr: bool = True
    min_lr: float = 4e-5

    ########### batch size & step ###########
    global_batch_size: int = 320
    micro_batch_size: int = 20
    max_step: int = None
    stop_step: int = None
    max_train_tokens: int = 3 * 10**12
    block_size: int = 2048  # TODO: this is config dependent; hardcoding it here
    warmup_steps: int = 2000
    log_step_interval: int = 10
    eval_iters: int = 100
    save_step_interval: int = 5000
    eval_step_interval: int = 5000
    log_train_loss_per_batch: bool = False
    activation_checkpointing: bool = False  # NOTE: xformers.swiglu is incompatible

    ########### seed and paths ###########
    random_seed: int = 3407
    model_name: str = "tiny_LLaMA_1b"
    tokenizer_path: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
    checkpoint_path: str = "checkpoints"
    # wandb
    wandb_dir = "./wandb"
    wandb_entity = "syncdoth"
    wandb_project = "LIT Training"

    ########## copy experiment ###########
    copy_exp: bool = False
    freeze_copied_weights: bool = False
    skip_reten: bool = True
    skip_oproj: bool = False
    skip_mlp: bool = False
    copy_model_path: str = "TinyLlama-1.1B-3T/NucleusX/lit_model-skip_reten.pth"
    copy_ckpt_dir: Path = None  # "RetNet-410m-bs1024-pile_dedup-copy_exp-skip_reten-freeze"
    copy_unfreeze_from: int | str = "latest"

    ########## Block Diagonal Attention ###########
    use_bda: bool = False

    ### datset option #####
    dataset: str = "refinedweb"

    ### Quick ad-hoc solution to old ckpt with wrong group norm ####
    use_ln_for_groupnorm: bool = False

    def __post_init__(self):
        batch_size = self.global_batch_size // self.num_of_devices
        self.gradient_accumulation_steps = batch_size // self.micro_batch_size
        assert self.gradient_accumulation_steps > 0
        self.warmup_iters = self.warmup_steps * self.gradient_accumulation_steps

        if self.max_step is None:
            self.max_step = math.ceil(
                self.max_train_tokens /
                (self.global_batch_size * self.num_of_nodes * self.block_size))

        self.max_iters = self.max_step * self.gradient_accumulation_steps
        self.stop_iters = self.stop_step * self.gradient_accumulation_steps if self.stop_step else self.max_iters
        self.lr_decay_iters = self.max_iters
        self.log_iter_interval = self.log_step_interval * self.gradient_accumulation_steps

        # paths
        self.name = f"{self.model_name}-bs{self.global_batch_size * self.num_of_nodes}-{self.dataset}"
        if self.copy_exp:
            self.name += "-copy_exp"
            if self.skip_reten:
                self.name += "-skip_reten"
            if self.skip_oproj:
                self.name += "-skip_oproj"
            if self.skip_mlp:
                self.name += "-skip_mlp"
            if self.freeze_copied_weights:
                self.name += "-freeze"
            if self.copy_ckpt_dir:
                self.copy_ckpt_dir = Path(self.copy_ckpt_dir)
                self.name += f"-unfreeze_from-{self.copy_unfreeze_from}"
        if self.use_bda:
            self.name += "-bda"
        self.out_dir = Path(self.checkpoint_path) / self.name

        # dataset
        if self.dataset in data_class_map:
            self.train_data_config = [(self.dataset, 1)]
        elif self.dataset == "reproduce":
            # Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
            self.train_data_config = [
                ("train_slim", 0.693584),
                ("train_star", 0.306416),
            ]

        self.val_data_config = [
            # ("validation", 1.0),  # NOTE: no validation
        ]


def get_loggers(name, log_iter_interval, wandb_dir=None, wandb_entity=None, wandb_project=None):
    logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_iter_interval)
    wandb_logger = WandbLogger(
        name=name,
        dir=wandb_dir,
        project=wandb_project,
        entity=wandb_entity,
    )
    return [logger, wandb_logger]


def setup(
    devices: int = 8,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
    num_of_nodes: int = 1,
    model_name: str = "tiny_LLaMA_1b",
    random_seed: int = 3407,
    micro_batch_size: int = 16,
    global_batch_size: int = 256,
    dataset: str = "refinedweb",
    learning_rate: float = 4e-4,
    min_lr: float = 4e-5,
    copy_exp: bool = False,
    freeze_copied_weights: bool = False,
    tokenizer_path: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
    optimizer_choice: str = "adamw",
    use_bda: bool = False,
    activation_checkpointing: bool = False,
    max_step: int = None,
    stop_step: int = None,
    use_ddp: bool = False,
    **kwargs,
) -> None:
    hparams = HyperParameters(num_of_devices=devices,
                              num_of_nodes=num_of_nodes,
                              model_name=model_name,
                              random_seed=random_seed,
                              micro_batch_size=micro_batch_size,
                              global_batch_size=global_batch_size,
                              dataset=dataset,
                              learning_rate=learning_rate,
                              min_lr=min_lr,
                              copy_exp=copy_exp,
                              freeze_copied_weights=freeze_copied_weights,
                              tokenizer_path=tokenizer_path,
                              optimizer_choice=optimizer_choice,
                              use_bda=use_bda,
                              activation_checkpointing=activation_checkpointing,
                              max_step=max_step,
                              stop_step=stop_step,
                              **kwargs)
    loggers = get_loggers(hparams.name,
                          hparams.log_iter_interval,
                          wandb_dir=hparams.wandb_dir,
                          wandb_entity=hparams.wandb_entity,
                          wandb_project=hparams.wandb_project)
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        elif use_ddp:
            strategy = DDPStrategy()
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy={Block}
                if hparams.activation_checkpointing else None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
                use_orig_params=True,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=loggers)
    fabric.print(hparams)
    if num_of_nodes > 1:
        main(fabric, train_data_dir, val_data_dir, resume, hparams)
    else:
        fabric.launch(main, train_data_dir, val_data_dir, resume, hparams)


def main(fabric, train_data_dir, val_data_dir, resume, hparams):
    monitor = Monitor(fabric,
                      window_size=2,
                      time_unit="seconds",
                      log_iter_interval=hparams.log_iter_interval)
    tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_path)

    if fabric.global_rank == 0:
        hparams.out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(hparams.model_name,
                              _eos_token_id=tokenizer.eos_token_id,
                              use_bda=hparams.use_bda)
    if config.block_size != hparams.block_size:
        fabric.print(
            f"Overriding config.block_size ({config.block_size}) with hparams.block_size ({hparams.block_size})"
        )
        config.block_size = hparams.block_size

    if config.use_ln_for_groupnorm != hparams.use_ln_for_groupnorm:
        fabric.print(
            f"Overriding config.use_ln_for_groupnorm to {hparams.use_ln_for_groupnorm}"
        )
        config.use_ln_for_groupnorm = hparams.use_ln_for_groupnorm

    fabric.seed_everything(hparams.random_seed)  # same seed for every process to init model (FSDP)
    random.seed(hparams.random_seed)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

        if hparams.copy_exp and hparams.copy_ckpt_dir is None:
            # state_dict = torch.load(hparams.copy_model_path, map_location="cpu")
            # if hparams.skip_reten:
            #     state_dict = {k: v for k, v in state_dict.items() if "attn.reten" not in k}
            # load_status = model.load_state_dict(state_dict, strict=False)
            # expected_missing_keys = [
            #     "attn.group_norm.weight",
            #     "attn.gate_proj.weight",
            # ]
            # if hparams.skip_reten:
            #     expected_missing_keys.append("attn.reten.weight")
            # assert all(any(x in key for x in expected_missing_keys) for key in load_status.missing_keys), load_status.missing_keys
            # assert len(load_status.unexpected_keys) == 0, load_status.unexpected_keys
            fabric.load_raw(hparams.copy_model_path, model, strict=False)

            torch.cuda.empty_cache()

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    if hparams.copy_exp:
        freeze_weights = ["wte", "ln_f", "lm_head"]
        if not hparams.skip_mlp:
            freeze_weights.extend(["mlp", "norm_2"])
        if not hparams.skip_oproj:
            freeze_weights.extend(["attn.proj", "norm_1"])
        if not hparams.skip_reten:
            freeze_weights.append("attn.reten")

    if hparams.freeze_copied_weights:
        for n, p in model.named_parameters():
            if any(fw in n for fw in freeze_weights):
                p.requires_grad = False

    model = fabric.setup(model)
    if hparams.copy_ckpt_dir is not None and not resume:
        params_to_train = [
            p for n, p in model.named_parameters() if not any(fw in n for fw in freeze_weights)
        ]
        extra_params_to_train = [
            p for n, p in model.named_parameters() if any(fw in n for fw in freeze_weights)
        ]
    else:
        params_to_train = [p for p in model.parameters() if p.requires_grad]

    if hparams.optimizer_choice == "adamw":
        optimizer = torch.optim.AdamW(params_to_train,
                                      lr=hparams.learning_rate,
                                      weight_decay=hparams.weight_decay,
                                      betas=(hparams.beta1, hparams.beta2),
                                      foreach=False)

    elif hparams.optimizer_choice == "lion":
        optimizer = Lion(
            params_to_train,
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
            betas=(hparams.beta1, hparams.beta2),
            use_triton=False,
        )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams.__dict__,
        "iter_num": 0,
        "step_count": 0
    }
    unfreeze = False
    if hparams.copy_ckpt_dir and not resume:
        if hparams.copy_unfreeze_from == "latest":
            unfreeze = sorted(hparams.copy_ckpt_dir.glob("*.pth"))[-1]
        else:
            unfreeze = hparams.copy_ckpt_dir / f"iter-{hparams.copy_unfreeze_from:06d}-ckpt.pth"
        fabric.print(f"Loading model from {unfreeze}")
        fabric.load(unfreeze, state)

        optimizer.add_param_group({"params": extra_params_to_train})

        torch.cuda.empty_cache()

    if resume is True:
        resume = sorted(hparams.out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
        torch.cuda.empty_cache()

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=hparams.micro_batch_size,  # * hparams.num_of_devices * hparams.num_of_nodes
        block_size=config.block_size,
        fabric=fabric,
        hparams=hparams,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        samples_to_skip=state["iter_num"],  # TODO: this is not accurate
        seed=hparams.random_seed,
        tokenizer=tokenizer,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader,
                                                                    val_dataloader)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume, unfreeze, hparams)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume, unfreeze, hparams):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader, hparams)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * hparams.micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (hparams.micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = 0

    loss_func = FusedCrossEntropyLoss()
    for train_data in train_dataloader:
        # train_data = fabric.broadcast(train_data)
        # train_data = train_data.to(fabric.device)
        # train_data = train_data[fabric.global_rank * hparams.micro_batch_size: (fabric.global_rank + 1) * hparams.micro_batch_size]

        # # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        # if resume:
        #     if curr_iter < initial_iter:
        #         curr_iter += 1
        #         continue
        #     else:
        #         resume = False
        #         curr_iter = -1
        #         fabric.barrier()
        #         fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= min(hparams.max_iters, hparams.stop_iters):
            break

        # determine and set the learning rate for this iteration
        if hparams.decay_lr:
            lr = get_lr(state["iter_num"], hparams.learning_rate, hparams.warmup_iters,
                        hparams.lr_decay_iters, hparams.min_lr)
            if unfreeze:
                # NOTE: these weights are newly added, and having them continue
                # with high learning rate may blow up the loss. Need to add a
                # separate warmup step for them.
                if state["iter_num"] < initial_iter + hparams.warmup_iters:
                    max_lr = get_lr(initial_iter + hparams.warmup_iters, hparams.learning_rate,
                                    hparams.warmup_iters, hparams.lr_decay_iters, hparams.min_lr)
                    lr_for_unfrozen_params = get_lr(state["iter_num"] - initial_iter, max_lr,
                                                    hparams.warmup_iters,
                                                    hparams.lr_decay_iters - initial_iter,
                                                    hparams.min_lr)
                else:
                    lr_for_unfrozen_params = lr
        else:
            lr = hparams.learning_rate

        if unfreeze:
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr_for_unfrozen_params
            assert len(optimizer.param_groups) == 2
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0:model.config.block_size].contiguous()
        targets = train_data[:, 1:model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % hparams.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            fabric.backward(loss / hparams.gradient_accumulation_steps)

        gathered_loss = fabric.all_reduce(loss, reduce_op="mean")
        if not is_accumulating:
            grad_norm = fabric.clip_gradients(model, optimizer, max_norm=hparams.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
            grad_norm = fabric.all_reduce(grad_norm, reduce_op="mean")
            fabric.log_dict(
                {
                    "loss": gathered_loss.item(),
                    "lr": lr,
                    "grad_norm": grad_norm,
                },
                state["step_count"],
            )
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        # input_id: B L
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
            f"iter {state['iter_num']}/{hparams.stop_iters} step {state['step_count']}/{hparams.stop_step}: loss {gathered_loss.item():.4f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (hparams.stop_iters - state['iter_num']) / 3600:.2f} hours. "
            # print days as well
            f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (hparams.stop_iters - state['iter_num']) / 3600 / 24:.2f} days. "
        )

        if hparams.log_train_loss_per_batch:
            fabric.log_dict(
                {
                    "train_loss_per_batch": gathered_loss.item(),
                },
                state["iter_num"],
            )

        monitor.on_train_batch_end(
            state["iter_num"] * hparams.micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=gathered_loss.item())

        if val_dataloader is not None and not is_accumulating and state[
                "step_count"] % hparams.eval_step_interval == 0:

            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, hparams)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(
                f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict(
                {
                    "metric/val_loss":
                        val_loss.item(),
                    "total_tokens":
                        model.config.block_size *
                        (state["iter_num"] + 1) * hparams.micro_batch_size * fabric.world_size
                }, state["step_count"])
            fabric.log_dict(
                {
                    "metric/val_ppl":
                        math.exp(val_loss.item()),
                    "total_tokens":
                        model.config.block_size *
                        (state["iter_num"] + 1) * hparams.micro_batch_size * fabric.world_size
                }, state["step_count"])
            fabric.barrier()
        if not is_accumulating and state["step_count"] % hparams.save_step_interval == 0:
            checkpoint_path = hparams.out_dir / f"iter-{state['step_count']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader,
             hparams) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(hparams.eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= hparams.eval_iters:
            break
        input_ids = val_data[:, 0:model.config.block_size].contiguous()
        targets = val_data[:, 1:model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        losses[k] = loss.item()

    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric,
    hparams,
    shuffle: bool = True,
    seed: int = 12345,
    samples_to_skip: int = 0,
    split="train",
    tokenizer=None,
) -> DataLoader:
    datasets = []
    data_config = hparams.train_data_config if split == "train" else hparams.val_data_config
    for prefix, _ in data_config:
        # filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        # random.seed(seed)
        # random.shuffle(filenames)

        #     dataset = PackedDataset(
        #         filenames,
        #         # n_chunks control the buffer size.
        #         # Note that the buffer size also impacts the random shuffle
        #         # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
        #         n_chunks=8,
        #         block_size=block_size,
        #         shuffle=shuffle,
        #         seed=seed+fabric.global_rank,
        #         num_processes=fabric.world_size,
        #         process_rank=fabric.global_rank,
        #     )

        dataset = data_class_map[prefix](
            block_size,
            tokenizer,
            seed=seed + fabric.global_rank,
            add_bos=True,
            shuffle=shuffle,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
            samples_to_skip=samples_to_skip,
            split=split,
        )
        datasets.append(dataset)

    # if not datasets:
    #     raise RuntimeError(
    #         f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
    #     )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    hparams,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    samples_to_skip: int = 0,
    seed: int = 12345,
    tokenizer=None,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        hparams=hparams,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        samples_to_skip=samples_to_skip,
        split="train",
        tokenizer=tokenizer,
    )
    val_dataloader = (create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        hparams=hparams,
        data_dir=val_data_dir,
        shuffle=False,
        seed=seed,
        split="validation",
        tokenizer=tokenizer,
    ) if val_data_dir else None)
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision(TORCH_MATMUL_PRECISION)

    CLI(setup)
