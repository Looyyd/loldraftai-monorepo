# utils/match_dataset.py
import os
import glob
import torch
import random
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq

from utils.column_definitions import COLUMNS, ColumnType
from utils import PARQUET_READER_BATCH_SIZE
from utils.task_definitions import TASKS, TaskType


class MatchDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        transform=None,
        mask_champions=0.0,
        unknown_champion_id=None,
        task_stats=None,
    ):
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        self.task_stats = task_stats
        self.transform = transform
        self.total_samples = self._count_total_samples()
        self.mask_champions = mask_champions
        self.unknown_champion_id = unknown_champion_id
        # Shuffle the data files
        random.seed(42)  # For reproducibility
        random.shuffle(self.data_files)

    def _count_total_samples(self):
        total = 0
        for file_path in self.data_files:
            parquet_file = pq.ParquetFile(file_path)
            total += parquet_file.metadata.num_rows
        return total

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            iter_start = 0
            iter_end = len(self.data_files)
        else:
            # In a worker process
            per_worker = int(len(self.data_files) / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for file_path in self.data_files[iter_start:iter_end]:
            # Read the Parquet file in batches using PyArrow
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(
                batch_size=PARQUET_READER_BATCH_SIZE
            ):
                df_chunk = batch.to_pandas()
                # Shuffle the DataFrame
                df_chunk = df_chunk.sample(frac=1).reset_index(drop=True)
                for _, row in df_chunk.iterrows():
                    sample = self._get_sample(row)
                    if sample:
                        yield sample

    def _get_sample(self, row):
        sample = {}
        for col, col_def in COLUMNS.items():
            if col_def.column_type == ColumnType.CATEGORICAL:
                # Store as integer
                sample[col] = int(row[col])
            elif col_def.column_type == ColumnType.NUMERICAL:
                # Store as float
                sample[col] = float(row[col])
            elif col_def.column_type == ColumnType.LIST:
                if col == "champion_ids":
                    champion_ids = [int(ch_id) for ch_id in row[col]]
                # Apply masking
                if self.mask_champions > 0 and self.unknown_champion_id is not None:
                    champion_ids = [
                        (
                            ch_id
                            if random.random() > self.mask_champions
                            else self.unknown_champion_id
                        )
                        for ch_id in champion_ids
                    ]
                sample[col] = torch.tensor(champion_ids, dtype=torch.long)
            else:
                sample[col] = torch.tensor(row[col], dtype=torch.long)

        # Extract labels for all tasks
        for task_name in TASKS.keys():
            task_label = row[task_name]
            if TASKS[task_name].task_type == TaskType.BINARY_CLASSIFICATION:
                sample[task_name] = float(task_label)
            elif TASKS[task_name].task_type == TaskType.REGRESSION:
                # TODO: maybe move normalization of tasks and of features to the same place? collate function?
                # Normalize the regression target
                mean = self.task_stats["means"][task_name]
                std = self.task_stats["stds"][task_name]
                if std != 0:
                    normalized_label = (float(task_label) - mean) / std
                else:
                    normalized_label = float(task_label) - mean
                sample[task_name] = normalized_label

            elif TASKS[task_name].task_type == TaskType.MULTICLASS_CLASSIFICATION:
                sample[task_name] = int(task_label)

        if self.transform:
            sample = self.transform(sample)

        return sample
