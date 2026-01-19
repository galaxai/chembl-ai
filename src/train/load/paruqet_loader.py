from collections.abc import Sequence
from pathlib import Path
from typing import Generator

import numpy as np
import pyarrow.dataset as ds
from tinygrad.tensor import Tensor


class ParquetDataLoader:
    def __init__(
        self,
        dir: str,
        X: str,
        Y: str,
        batch_size: int = 512,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.dir = dir
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle

        Tensor.manual_seed(self.seed)

        ## Load the data
        self.data = list(Path(self.dir).glob("*.parquet"))
        if not self.data:
            from os.path import abspath

            raise FileNotFoundError(
                f"No parquet files found in {self.dir}. Directory {abspath(self.dir)}"
            )

    def __iter__(self) -> Generator[tuple[Tensor, Tensor], None, None]:
        if self.shuffle:
            indices_raw = Tensor.randperm(len(self.data)).tolist()
            if isinstance(indices_raw, Sequence):
                indices = [int(i) for i in indices_raw]
            else:
                indices = [int(indices_raw)]
            files = [self.data[i] for i in indices]
        else:
            files = self.data
        data = ds.dataset(files)

        # TODO parametrize zero_copy_only or write helper function
        for batch in data.to_batches(batch_size=self.batch_size):
            x = Tensor(np.stack(batch[self.X].to_numpy(zero_copy_only=False)))
            y = Tensor(batch[self.Y].to_numpy(zero_copy_only=True))
            yield (x, y)
