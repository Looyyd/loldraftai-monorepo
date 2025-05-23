# utils/match_prediction/match_dataset.py
import os
import glob
import random
import pickle
import torch
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from torch.utils.data import IterableDataset
from typing import Callable, Optional, List, Dict, Any
import psutil
import multiprocessing
import gc

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
            "num_workers": 8,  # Increased back to 8 since GPU is faster now
            "prefetch_factor": 16,  # Increased significantly for deeper pipeline
            "pin_memory": True,
            "persistent_workers": True,
        }
    elif device.type == "mps":
        # M1/M2 Mac config
        return {
            "num_workers": 2,  # Increased for M1/M2
            "prefetch_factor": 8,  # Increased
            "pin_memory": True,
            "persistent_workers": True,
        }
    else:  # CPU
        print("Using CPU dataloader config")
        # For 4 CPU machine, reserve 1 CPU for main process
        total_cpus = multiprocessing.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Use more workers for CPU training to compensate
        return {
            "num_workers": min(total_cpus - 1, 6),  # Increased workers
            "prefetch_factor": (
                4 if available_memory_gb > 8 else 2
            ),  # Increased prefetch
            "pin_memory": False,  # False for CPU training
            "persistent_workers": True,
        }


# Use in DataLoader initialization
dataloader_config = get_dataloader_config()


class OptimizedMatchDataset(IterableDataset):
    def __init__(
        self,
        masking_function: Optional[Callable[[], int]] = None,
        unknown_champion_id=None,
        train_or_test="train",
        dataset_fraction: float = 1.0,
        reshuffle_fraction: bool = True,
        batch_size: int = 512,  # Internal batch size for processing
    ):
        self.data_files = sorted(
            glob.glob(os.path.join(PREPARED_DATA_DIR, train_or_test, f"*.parquet"))
        )
        self.dataset_fraction = dataset_fraction
        self.train_or_test = train_or_test
        self.reshuffle_fraction = reshuffle_fraction
        self.all_data_files = self.data_files.copy()
        self.batch_size = batch_size

        # Initial file selection
        if dataset_fraction < 1.0:
            num_files = max(1, int(len(self.data_files) * dataset_fraction))
            self.data_files = self.data_files[:num_files]

        self.total_samples = self._count_total_samples()
        self.masking_function = masking_function
        self.unknown_champion_id = unknown_champion_id

        # Shuffle the data files once
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
            try:
                parquet_file = pq.ParquetFile(file_path)
                total += parquet_file.metadata.num_rows
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")
                continue

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
            try:
                # Use context manager to ensure file is closed
                with pq.ParquetFile(file_path) as parquet_file:
                    for batch in parquet_file.iter_batches(
                        batch_size=PARQUET_READER_BATCH_SIZE
                    ):
                        # Convert to pandas once
                        df_chunk = batch.to_pandas()

                        if len(df_chunk) == 0:
                            continue

                        # Shuffle the chunk once
                        df_chunk = df_chunk.sample(frac=1).reset_index(drop=True)

                        # Process the chunk and yield individual samples
                        yield from self._process_chunk_optimized(df_chunk)

                        # Clean up memory
                        del df_chunk

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

    def _process_chunk_optimized(self, df_chunk: pd.DataFrame):
        """Optimized chunk processing with vectorized operations"""
        # Handle champion masking vectorized if needed
        if self.masking_function is not None and self.unknown_champion_id is not None:
            df_chunk = self._apply_champion_masking_vectorized(df_chunk)

        # Convert champion_ids to tensors vectorized
        champion_ids_tensors = [
            torch.tensor(x, dtype=torch.long) for x in df_chunk["champion_ids"].values
        ]
        df_chunk["champion_ids"] = champion_ids_tensors

        # Convert to records efficiently
        # Instead of to_dict("records"), use itertuples which is much faster
        for row in df_chunk.itertuples(index=False, name=None):
            # Convert namedtuple to dict manually (faster than to_dict)
            sample = {col: row[i] for i, col in enumerate(df_chunk.columns)}
            yield sample

    def _apply_champion_masking_vectorized(
        self, df_chunk: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply champion masking using vectorized operations"""

        def mask_champions_vectorized(champion_ids_series, num_to_mask_per_row):
            def mask_single_row(champion_list, num_to_mask):
                if num_to_mask == 0 or len(champion_list) == 0:
                    return champion_list
                mask_indices = np.random.choice(
                    len(champion_list),
                    size=min(num_to_mask, len(champion_list)),
                    replace=False,
                )
                return [
                    self.unknown_champion_id if i in mask_indices else ch_id
                    for i, ch_id in enumerate(champion_list)
                ]

            # Apply masking to each row
            if isinstance(num_to_mask_per_row, int):
                # Same number for all rows
                return champion_ids_series.apply(
                    lambda x: mask_single_row(x, num_to_mask_per_row)
                )
            else:
                # Different number per row
                return champion_ids_series.apply(
                    lambda x: mask_single_row(x, self.masking_function())
                )

        df_chunk = df_chunk.copy()
        num_to_mask = self.masking_function()
        df_chunk["champion_ids"] = mask_champions_vectorized(
            df_chunk["champion_ids"], num_to_mask
        )
        return df_chunk


class BatchedMatchDataset(IterableDataset):
    """Even more optimized version that yields pre-batched data"""

    def __init__(
        self,
        masking_function: Optional[Callable[[], int]] = None,
        unknown_champion_id=None,
        train_or_test="train",
        dataset_fraction: float = 1.0,
        reshuffle_fraction: bool = True,
        target_batch_size: int = 512,
    ):
        self.data_files = sorted(
            glob.glob(os.path.join(PREPARED_DATA_DIR, train_or_test, f"*.parquet"))
        )
        self.dataset_fraction = dataset_fraction
        self.train_or_test = train_or_test
        self.reshuffle_fraction = reshuffle_fraction
        self.all_data_files = self.data_files.copy()
        self.target_batch_size = target_batch_size

        # Initial file selection
        if dataset_fraction < 1.0:
            num_files = max(1, int(len(self.data_files) * dataset_fraction))
            self.data_files = self.data_files[:num_files]

        self.total_samples = self._count_total_samples()
        self.masking_function = masking_function
        self.unknown_champion_id = unknown_champion_id

        # Pre-compute column info for faster processing
        self._categorical_columns = None
        self._float_columns = None
        self._initialize_column_info()

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

        # Fall back to counting
        print(f"Counting samples in {self.train_or_test} dataset...")
        total = 0
        for file_path in self.data_files[:10]:  # Sample first 10 files for speed
            try:
                parquet_file = pq.ParquetFile(file_path)
                total += parquet_file.metadata.num_rows
            except Exception:
                continue

        # Estimate total from sample
        total = total * len(self.data_files) // min(10, len(self.data_files))
        return int(total * self.dataset_fraction)

    def _initialize_column_info(self):
        """Initialize column type information for faster processing"""
        if not self.data_files:
            return

        try:
            # Sample first file to get column info
            with pq.ParquetFile(self.data_files[0]) as pf:
                sample_batch = next(pf.iter_batches(batch_size=1))
                df_sample = sample_batch.to_pandas()

                self._categorical_columns = []
                self._float_columns = []

                for col in df_sample.columns:
                    if col == "champion_ids":
                        continue  # Special handling
                    elif df_sample[col].dtype in ["int32", "int64"]:
                        self._categorical_columns.append(col)
                    elif df_sample[col].dtype in ["float32", "float64"]:
                        self._float_columns.append(col)

        except Exception as e:
            print(f"Warning: Could not initialize column info: {e}")
            self._categorical_columns = []
            self._float_columns = []

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # File selection logic
        if self.dataset_fraction < 1.0 and self.reshuffle_fraction:
            files = self.all_data_files.copy()
            random.shuffle(files)
            num_files = max(1, int(len(files) * self.dataset_fraction))
            files = files[:num_files]
        else:
            files = self.data_files.copy()

        random.shuffle(files)

        if worker_info is None:
            iter_start, iter_end = 0, len(files)
        else:
            per_worker = int(np.ceil(len(files) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(files))

        sample_buffer = []

        for file_path in files[iter_start:iter_end]:
            try:
                with pq.ParquetFile(file_path) as parquet_file:
                    for batch in parquet_file.iter_batches(
                        batch_size=PARQUET_READER_BATCH_SIZE
                    ):
                        df_chunk = batch.to_pandas()

                        if len(df_chunk) == 0:
                            continue

                        # Shuffle chunk
                        df_chunk = df_chunk.sample(frac=1).reset_index(drop=True)

                        # Process chunk to samples and add to buffer
                        samples = self._chunk_to_samples_fast(df_chunk)
                        sample_buffer.extend(samples)

                        # Yield batches when buffer is full
                        while len(sample_buffer) >= self.target_batch_size:
                            batch_samples = sample_buffer[: self.target_batch_size]
                            sample_buffer = sample_buffer[self.target_batch_size :]

                            for sample in batch_samples:
                                yield sample

                        # Clean up
                        del df_chunk, samples

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        # Yield remaining samples
        for sample in sample_buffer:
            yield sample

    def _chunk_to_samples_fast(self, df_chunk: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert chunk to samples with optimized processing"""
        # Apply champion masking if needed
        if self.masking_function is not None and self.unknown_champion_id is not None:
            num_to_mask = self.masking_function()
            if num_to_mask > 0:
                df_chunk = df_chunk.copy()
                df_chunk["champion_ids"] = df_chunk["champion_ids"].apply(
                    lambda x: self._mask_champions_fast(x, num_to_mask)
                )

        # Convert champion_ids to tensors
        champion_tensors = [
            torch.tensor(x, dtype=torch.long) for x in df_chunk["champion_ids"].values
        ]

        # Build samples efficiently
        samples = []
        for i, (_, row) in enumerate(df_chunk.iterrows()):
            sample = row.to_dict()
            sample["champion_ids"] = champion_tensors[i]
            samples.append(sample)

        return samples

    def _mask_champions_fast(
        self, champion_list: List[int], num_to_mask: int
    ) -> List[int]:
        """Fast champion masking"""
        if num_to_mask == 0 or len(champion_list) == 0:
            return champion_list

        mask_indices = np.random.choice(
            len(champion_list), size=min(num_to_mask, len(champion_list)), replace=False
        )

        return [
            self.unknown_champion_id if i in mask_indices else ch_id
            for i, ch_id in enumerate(champion_list)
        ]


# Use the optimized version as default
MatchDataset = OptimizedMatchDataset


# Add this new optimized dataset class
class HighThroughputMatchDataset(IterableDataset):
    """Ultra-optimized dataset for maximum GPU utilization"""

    def __init__(
        self,
        masking_function: Optional[Callable[[], int]] = None,
        unknown_champion_id=None,
        train_or_test="train",
        dataset_fraction: float = 1.0,
        reshuffle_fraction: bool = True,
    ):
        self.data_files = sorted(
            glob.glob(os.path.join(PREPARED_DATA_DIR, train_or_test, f"*.parquet"))
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

        # Shuffle the data files once
        random.seed(42)
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

        # Fast estimation using first few files
        total = 0
        sample_files = self.data_files[: min(5, len(self.data_files))]
        for file_path in sample_files:
            try:
                parquet_file = pq.ParquetFile(file_path)
                total += parquet_file.metadata.num_rows
            except Exception:
                continue

        # Estimate total
        if sample_files:
            total = total * len(self.data_files) // len(sample_files)

        return int(total * self.dataset_fraction)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # File selection logic
        if self.dataset_fraction < 1.0 and self.reshuffle_fraction:
            files = self.all_data_files.copy()
            random.shuffle(files)
            num_files = max(1, int(len(files) * self.dataset_fraction))
            files = files[:num_files]
        else:
            files = self.data_files.copy()

        random.shuffle(files)

        if worker_info is None:
            iter_start, iter_end = 0, len(files)
        else:
            per_worker = int(np.ceil(len(files) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(files))

        # Use larger read buffer for better I/O performance
        read_buffer_size = PARQUET_READER_BATCH_SIZE * 2

        for file_path in files[iter_start:iter_end]:
            try:
                with pq.ParquetFile(file_path) as parquet_file:
                    for batch in parquet_file.iter_batches(batch_size=read_buffer_size):
                        df_chunk = batch.to_pandas()

                        if len(df_chunk) == 0:
                            continue

                        # Shuffle once
                        df_chunk = df_chunk.sample(frac=1).reset_index(drop=True)

                        # Process entire chunk at once for efficiency
                        yield from self._process_chunk_ultra_fast(df_chunk)

                        del df_chunk

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

    def _process_chunk_ultra_fast(self, df_chunk: pd.DataFrame):
        """Ultra-fast chunk processing with minimal overhead"""
        # Apply masking if needed - simplified approach
        if self.masking_function is not None and self.unknown_champion_id is not None:
            num_to_mask = self.masking_function()
            if num_to_mask > 0:
                # Apply masking directly without nested functions
                df_chunk = df_chunk.copy()
                df_chunk["champion_ids"] = df_chunk["champion_ids"].apply(
                    lambda x: self._mask_champions_simple(x, num_to_mask)
                )

        # Convert to tensors in batch - much faster than per-sample
        champion_tensors = [
            torch.tensor(x, dtype=torch.long) for x in df_chunk["champion_ids"].values
        ]

        # Use numpy arrays for faster iteration
        column_names = list(df_chunk.columns)
        data_arrays = [df_chunk[col].values for col in column_names]

        # Generate samples with minimal overhead
        for i in range(len(df_chunk)):
            sample = {}
            for j, col in enumerate(column_names):
                if col == "champion_ids":
                    sample[col] = champion_tensors[i]
                else:
                    sample[col] = data_arrays[j][i]
            yield sample

    def _mask_champions_simple(
        self, champion_list: List[int], num_to_mask: int
    ) -> List[int]:
        """Simple champion masking - must be picklable"""
        if num_to_mask == 0 or len(champion_list) == 0:
            return champion_list

        mask_indices = np.random.choice(
            len(champion_list), size=min(num_to_mask, len(champion_list)), replace=False
        )

        result = champion_list.copy()
        for idx in mask_indices:
            result[idx] = self.unknown_champion_id
        return result


# Use the high-throughput version as default for training
MatchDataset = HighThroughputMatchDataset
