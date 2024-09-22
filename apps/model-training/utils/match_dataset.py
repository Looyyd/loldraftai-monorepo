# match_dataset.py
import os
import pickle
import glob

import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq


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

        # Load label encoders
        with open(label_encoders_path, "rb") as f:
            self.label_encoders = pickle.load(f)

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
        # Extract features
        region = row["region"]
        average_tier = row["averageTier"]
        average_division = row["averageDivision"]
        champion_ids = row["champion_ids"]  # This is a list

        # Label
        label = row["label"]

        sample = {
            "region": torch.tensor(region, dtype=torch.long),
            "averageTier": torch.tensor(average_tier, dtype=torch.long),
            "averageDivision": torch.tensor(average_division, dtype=torch.long),
            "champion_ids": torch.tensor(champion_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
