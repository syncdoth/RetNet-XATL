import glob

from tqdm import trange
import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset, concatenate_datasets


class StarCoderDataset(IterableDataset):
    separate_files = [
        "jupyter-scripts-dedup-filtered",
        "jupyter-structured-clean-dedup",
        "github-issues-filtered-structured",
        "git-commits-cleaned",
    ]

    def __init__(
        self,
        sequence_length,
        tokenizer,
        seed=42,
        add_bos=False,
        shuffle=True,
        samples_to_skip=0,
        num_processes=1,
        process_rank=0,
        split="train",
        data_dir="starcoderdata",
        cache_dir="data_cache/star_coder",
    ):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self._num_processes = num_processes
        self._process_rank = process_rank

        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.tokenizer.add_bos_token = add_bos
        self.buffer = []
        self.shuffle = shuffle
        self.seed = seed
        self.split = split

        data_files = sorted(glob.glob(f"{self.data_dir}/*/*.parquet", recursive=True))
        self.data_files1 = []
        self.data_files2 = []
        for fname in data_files:
            if any([y in fname for y in self.separate_files]):
                self.data_files2.append(fname)
            else:
                self.data_files1.append(fname)

        self.samples_to_skip = samples_to_skip
        # TODO: skipping is not well defined, since the logic for dataloader is
        # different from accelerate's distribute_batches=True.
        # shards_to_skip = samples_to_skip // self.len_shard
        # num_shards = len(data_files)
        # shards_to_skip = shards_to_skip % num_shards

        # data_files = data_files[shards_to_skip:]
        # samples_to_skip = samples_to_skip % self.len_shard

    def load_dataset(self, data_files1, data_files2, samples_to_skip=0):
        dataset1 = load_dataset(
            "parquet",
            data_files=data_files1,
            cache_dir=self.cache_dir,
            split="train",
            streaming=True,
        )
        dataset2 = load_dataset(
            "parquet",
            data_files=data_files2,
            cache_dir=self.cache_dir,
            split="train",
            streaming=True,
        )
        dataset1 = dataset1.remove_columns(
            [col for col in dataset1.column_names if col != "content"])
        dataset2 = dataset2.remove_columns(
            [col for col in dataset2.column_names if col != "content"])
        self.dataset = concatenate_datasets([dataset1, dataset2])

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
        return row["content"]

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

        max_num_files1 = len(self.data_files1) // num_shards * num_shards
        filenames1 = self.data_files1[shard_id:max_num_files1:num_shards]

        max_num_files2 = len(self.data_files2) // num_shards * num_shards
        filenames2 = self.data_files2[shard_id:max_num_files2:num_shards]
        data_iter = self.load_dataset(filenames1, filenames2, samples_to_skip=self.samples_to_skip)

        for row in data_iter:
            content = self.load_text(row)
            self.buffer += self.encode(content)
            while len(self.buffer) >= self.sequence_length:
                token_ids = torch.tensor(self.buffer[:self.sequence_length])
                self.buffer = self.buffer[self.sequence_length:]
                yield token_ids
