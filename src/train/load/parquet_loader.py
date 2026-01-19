import random
from pathlib import Path
from typing import Generator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from tinygrad.tensor import Tensor


class ParquetDataLoader:
    """Stream parquet rows in mini-batches as tinygrad tensors."""

    def __init__(
        self,
        dir: str,
        X: str,
        Y: str,
        batch_size: int = 512,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """Initialize loader settings and seed RNGs."""
        self.dir = dir
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        Tensor.manual_seed(self.seed)
        random.seed(self.seed)

        ## Load the data
        self.data = list(Path(self.dir).glob("*.parquet"))
        if not self.data:
            from os.path import abspath

            raise FileNotFoundError(
                f"No parquet files found in {self.dir}. Directory {abspath(self.dir)}"
            )

    def __iter__(self) -> Generator[tuple[Tensor, Tensor], None, None]:
        """Yield batches of feature/label tensors from parquet rows."""
        if self.shuffle:
            random.shuffle(self.data)
        data = ds.dataset(self.data)

        def batch_to_tensors(batch: pa.RecordBatch) -> tuple[Tensor, Tensor]:
            x = Tensor(np.stack(batch[self.X].to_numpy(zero_copy_only=False)))
            y = Tensor(batch[self.Y].to_numpy(zero_copy_only=True))
            return x, y

        remainder_batch = None
        for batch in data.to_batches(batch_size=self.batch_size):
            if remainder_batch is not None:
                batch = pa.concat_batches([remainder_batch, batch])
                remainder_batch = None

            while batch.num_rows >= self.batch_size:
                out = batch.slice(0, self.batch_size)
                batch = batch.slice(self.batch_size)
                yield batch_to_tensors(out)

            if batch.num_rows > 0:
                remainder_batch = batch

        if remainder_batch is not None and not self.drop_last:
            yield batch_to_tensors(remainder_batch)

    def __len__(self) -> int:
        """Return the number of batches implied by dataset size."""
        import math

        if not hasattr(self, "_num_batches"):
            data = ds.dataset(self.data)
            total_rows = data.count_rows()
            if self.drop_last:
                self._num_batches = total_rows // self.batch_size
            else:
                self._num_batches = math.ceil(total_rows / self.batch_size)
        return self._num_batches
