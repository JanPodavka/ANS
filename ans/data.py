import math
from typing import Iterator, Optional

import torch


class BatchLoader:

    def __init__(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            shuffle: bool = False
    ) -> None:
        """

        Args:
            x: The first dimension should correspond to batch index, i.e. x[i, ...] is the i-th sample
            y: In case of supervised problems (e.g. classification), e.g. y[i] should be the target for x[i]
            batch_size: How many samples in batch
            shuffle: If True, then the data should be randomly reordered on each __iter__
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size or len(x)
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """

        Returns:
            batch: If unsupervised (i.e. self.y is None), return single element tuple (x[batch_ids],). If supervised
                   (i.e. self.y is torch.Tensor), return the pair (x[batch_ids], y[batch_ids])
        """

        ########################################
        idx = torch.randperm(self.x.shape[0]) if self.shuffle else torch.arange(self.x.shape[0])
        i = 0
        while i < self.x.shape[0]:
            if self.y is None:
                yield self.x[idx[i:i + self.batch_size]],
            else:
                yield self.x[idx[i:i + self.batch_size]], self.y[idx[i:i + self.batch_size]]

            i += self.batch_size

        ########################################

    def __len__(self) -> int:
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}:\n" \
               f"    num_batches: {len(self)}\n" \
               f"    batch_shape: {(self.batch_size,) + tuple(self.x.shape[1:])}\n"
