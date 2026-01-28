from pathlib import Path

import pyarrow.dataset as ds
from torch import float32, int64, tensor
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data


class SMILESDataset(IterableDataset):
    def __init__(self, parquet_dir):
        self.parquet_dir = Path(parquet_dir)

        # Load all parquet files
        self.parquet_files = sorted(self.parquet_dir.glob("*.parquet"))

        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {parquet_dir}")
        self.num_rows = ds.dataset(self.parquet_files).count_rows()

    def __len__(self):
        return self.num_rows

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_files = self.parquet_files[worker_id::num_workers]
        if not worker_files:
            return

        dataset = ds.dataset(worker_files)
        schema = dataset.schema
        nf_idx = schema.get_field_index("node_features")
        ei_idx = schema.get_field_index("edge_index")
        y_idx = schema.get_field_index("pIC")

        for batch in dataset.to_batches():
            nf = batch.column(nf_idx)
            ei = batch.column(ei_idx)
            y = batch.column(y_idx)

            for i in range(batch.num_rows):
                x = tensor(nf[i].as_py(), dtype=float32)
                edge_index = tensor(ei[i].as_py(), dtype=int64).contiguous()
                y_val = tensor(y[i].as_py(), dtype=float32)
                yield Data(x=x, edge_index=edge_index, y=y_val)
