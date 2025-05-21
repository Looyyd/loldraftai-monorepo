# utils/match_prediction/match_dataset.py
import os
import glob
import random
import pickle
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
from typing import Callable, Optional, List
import psutil
import multiprocessing

from utils.match_prediction import (
    PREPARED_DATA_DIR,
    SAMPLE_COUNTS_PATH,
    PARQUET_READER_BATCH_SIZE,
)
from utils.match_prediction import get_best_device


def get_dataloader_config():
    device = get_best_device()
    if device.type == "cuda":
        return {
            "num_workers": 8,
            "prefetch_factor": 4,
            "pin_memory": True,
        }
    elif device.type == "mps":
        # M1/M2 Mac config
        return {
            "num_workers": 1,
            "prefetch_factor": 4,
            "pin_memory": True,
        }
    else:  # CPU
        # For 4 CPU machine, reserve 1 CPU for main process
        total_cpus = multiprocessing.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Use 2 workers for 4 CPU machine, leaving 2 CPUs for main process and OS
        # Adjust prefetch_factor based on available memory
        return {
            "num_workers": min(total_cpus - 2, 2),  # Use 2 workers on 4 CPU machine
            "prefetch_factor": (
                2 if available_memory_gb > 8 else 1
            ),  # Lower if memory constrained
            "pin_memory": False,  # False for CPU training
        }


# Use in DataLoader initialization
dataloader_config = get_dataloader_config()


class MatchDataset(IterableDataset):
    def __init__(
        self,
        masking_function: Optional[Callable[[], int]] = None,
        unknown_champion_id=None,
        train_or_test="train",
        dataset_fraction: float = 1.0,
        reshuffle_fraction: bool = True,
    ):
        self.data_files = sorted(
            glob.glob(
                os.path.join(
                    PREPARED_DATA_DIR, train_or_test, f"{train_or_test}*.parquet"
                )
            )
        )
        self.dataset_fraction = dataset_fraction
        self.train_or_test = train_or_test
        self.reshuffle_fraction = reshuffle_fraction
        self.all_data_files = self.data_files.copy()

        # Initial file selection
        if dataset_fraction < 1.0:
            num_files = max(1, int(len(self.data_files) * dataset_fraction))
            self.data_files = self.data_files[:num_files]

        self.total_samples = self._count_total_samples()
        self.masking_function = masking_function
        self.unknown_champion_id = unknown_champion_id

        # Shuffle the data files
        random.seed(42)  # For reproducibility
        random.shuffle(self.data_files)

    def _count_total_samples(self):
        try:
            if os.path.exists(SAMPLE_COUNTS_PATH):
                with open(SAMPLE_COUNTS_PATH, "rb") as f:
                    counts = pickle.load(f)
                    count = counts.get(self.train_or_test)
                    if count is not None:
                        return int(count * self.dataset_fraction)
        except Exception as e:
            print(f"Warning: Error reading sample counts file: {e}")

        # Fall back to counting if file doesn't exist or is invalid
        print(f"Counting samples in {self.train_or_test} dataset...")
        total = 0
        for file_path in self.data_files:
            parquet_file = pq.ParquetFile(file_path)
            total += parquet_file.metadata.num_rows

        # Try to save the count for future use
        try:
            counts = {}
            if os.path.exists(SAMPLE_COUNTS_PATH):
                with open(SAMPLE_COUNTS_PATH, "rb") as f:
                    counts = pickle.load(f)
            counts[self.train_or_test] = total
            with open(SAMPLE_COUNTS_PATH, "wb") as f:
                pickle.dump(counts, f)
        except Exception as e:
            print(f"Warning: Could not save sample counts: {e}")

        return int(total * self.dataset_fraction)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # If using fraction and reshuffle is enabled, select new subset of files
        if self.dataset_fraction < 1.0 and self.reshuffle_fraction:
            files = self.all_data_files.copy()
            random.shuffle(files)
            num_files = max(1, int(len(files) * self.dataset_fraction))
            files = files[:num_files]
        else:
            files = self.data_files.copy()

        random.shuffle(files)  # Shuffle the selected files

        if worker_info is None:
            iter_start, iter_end = 0, len(files)
        else:
            per_worker = int(np.ceil(len(files) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(files))

        for file_path in files[iter_start:iter_end]:
            # Use context manager to ensure file is closed
            with pq.ParquetFile(file_path) as parquet_file:
                for batch in parquet_file.iter_batches(
                    batch_size=PARQUET_READER_BATCH_SIZE
                ):
                    df_chunk = batch.to_pandas()
                    df_chunk = df_chunk.sample(frac=1).reset_index(drop=True)

                    # Process the chunk and yield individual samples
                    samples = self._get_samples(df_chunk)
                    for sample in samples:
                        yield sample

    def _get_samples(self, df_chunk):
        # Only handle champion masking if needed
        if self.masking_function is not None and self.unknown_champion_id is not None:
            df_chunk["champion_ids"] = df_chunk["champion_ids"].apply(
                lambda x: self._mask_champions(x, self.masking_function())
            )

        # Convert champion_ids to tensor(expected by collate_fn)
        df_chunk["champion_ids"] = df_chunk["champion_ids"].apply(
            lambda x: torch.tensor(x, dtype=torch.long)
        )

        samples = df_chunk.to_dict("records")
        return samples

    def _mask_champions(self, champion_list: List[int], num_to_mask: int) -> List[int]:
        """Masks a specific number of champions in the list"""
        mask_indices = np.random.choice(
            len(champion_list), size=num_to_mask, replace=False
        )
        return [
            self.unknown_champion_id if i in mask_indices else ch_id
            for i, ch_id in enumerate(champion_list)
        ]
