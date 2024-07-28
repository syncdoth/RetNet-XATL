import glob

from tqdm import trange
import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset


class PileDedupDataset(IterableDataset):
    len_shard = 174919
    data_dir = "the_pile_deduplicated/data"
    cache_dir = "data_cache/pile_dedup"

    def __init__(self,
                 sequence_length,
                 tokenizer,
                 seed=42,
                 add_bos=False,
                 shuffle=True,
                 samples_to_skip=0,
                 num_processes=1,
                 process_rank=0,
                 split="train"):
        self._num_processes = num_processes
        self._process_rank = process_rank

        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.tokenizer.add_bos_token = add_bos
        self.buffer = []
        self.shuffle = shuffle
        self.seed = seed

        self.data_files = sorted(glob.glob(f"{self.data_dir}/*.parquet"))
        self.samples_to_skip = samples_to_skip
        # TODO: skipping is not well defined, since the logic for dataloader is
        # different from accelerate's distribute_batches=True.
        # shards_to_skip = samples_to_skip // self.len_shard
        # num_shards = len(data_files)
        # shards_to_skip = shards_to_skip % num_shards

        # data_files = data_files[shards_to_skip:]
        # samples_to_skip = samples_to_skip % self.len_shard

    def load_dataset(self, data_files, samples_to_skip=0):
        self.dataset = load_dataset(
            "parquet",
            data_files=data_files,
            cache_dir=self.cache_dir,
            split="train",
            streaming=True,
        )
        if self.shuffle:
            ds = self.dataset.shuffle(seed=self.seed, buffer_size=10000)
        else:
            ds = self.dataset

        _iter = iter(ds)
        for _ in trange(samples_to_skip,
                        desc=f"skipping {samples_to_skip} of {self.__class__.__name__}"):
            next(_iter)

        return _iter

    def load_text(self, row):
        return row["text"]

    def encode(self, content):
        encoded = self.tokenizer.encode(content)
        encoded += [self.tokenizer.eos_token_id]
        return encoded

    def __len__(self):
        return self.max_pages

    def __iter__(self):
        # if self._process_rank != 0:
        #     while True:
        #         yield []
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self.data_files) // num_shards * num_shards
        filenames = self.data_files[shard_id:max_num_files:num_shards]
        data_iter = self.load_dataset(filenames, samples_to_skip=self.samples_to_skip)

        for row in data_iter:
            content = self.load_text(row)
            self.buffer += self.encode(content)
            while len(self.buffer) >= self.sequence_length:
                token_ids = torch.tensor(self.buffer[:self.sequence_length])
                self.buffer = self.buffer[self.sequence_length:]
                yield token_ids
