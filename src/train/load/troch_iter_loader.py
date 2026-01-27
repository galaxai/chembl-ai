from pathlib import Path

import pyarrow.dataset as ds
from torch import float16, tensor
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data


class SMILESDataset(IterableDataset):
    def __init__(self, parquet_dir):
        self.parquet_dir = Path(parquet_dir)

        # Load all parquet files
        parquet_files = list(self.parquet_dir.glob("*.parquet"))

        if not parquet_files:
            raise ValueError(f"No parquet files found in {parquet_dir}")

        self.data = ds.dataset(parquet_files)

    def __len__(self):
        return self.data.count_rows()

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        for batch_idx, batch in enumerate(self.data.to_batches()):
            if batch_idx % num_workers != worker_id:
                continue

            nf = batch.column(batch.schema.get_field_index("node_features"))
            ei = batch.column(batch.schema.get_field_index("edge_index"))
            y = batch.column(batch.schema.get_field_index("pIC"))

            for i in range(batch.num_rows):
                x = tensor(nf[i].as_py())
                edge_index = tensor(ei[i].as_py()).contiguous()
                y_val = tensor(y[i].as_py(), dtype=float16)
                yield Data(x=x, edge_index=edge_index, y=y_val)
