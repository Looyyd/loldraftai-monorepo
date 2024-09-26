# utils/match_dataset.py
import os
import glob
import pickle
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq
from utils.column_definitions import COLUMNS, ColumnType


class MatchDataset(IterableDataset):
    def __init__(self, data_dir: str, label_encoders_path: str, transform=None):
        """
        Args:
            data_dir (str): Directory containing the Parquet files.
            label_encoders_path (str): Path to the saved label encoders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        self.transform = transform
        self.total_samples = self._count_total_samples()

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
            for batch in parquet_file.iter_batches(batch_size=1000):
                df_chunk = batch.to_pandas()
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
                # Keep tensors for list columns
                sample[col] = torch.tensor(row[col], dtype=torch.long)
        # Ensure label is a float
        sample["label"] = float(row["label"])

        if self.transform:
            sample = self.transform(sample)

        return sample
