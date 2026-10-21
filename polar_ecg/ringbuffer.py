"""
High-performance 2D circular ring buffer for real-time pyqtgraph plotting.
"""

import numpy as np


class RingBuffer:
    """
    Fixed-capacity circular buffer storing (x, y) coordinates for fast,
    allocation-free visualization updates.
    """

    def __init__(self, capacity: int, dtype: np.dtype = np.float64):
        if capacity <= 0:
            raise ValueError(f"RingBuffer capacity must be positive, got {capacity}")
        self._capacity = int(capacity)
        self._dtype = dtype
        self._x = np.zeros(self._capacity, dtype=self._dtype)
        self._y = np.zeros(self._capacity, dtype=self._dtype)
        self._head = 0  # Points to next write slot
        self._count = 0  # Number of valid items written

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return min(self._count, self._capacity)

    def clear(self) -> None:
        self._head = 0
        self._count = 0

    def append(self, x: float, y: float) -> None:
        """Append a single (x, y) point."""
        self._x[self._head] = x
        self._y[self._head] = y
        self._head = (self._head + 1) % self._capacity
        self._count += 1

    def append_many(self, x: np.ndarray, y: np.ndarray) -> None:
        """Append arrays of x and y points efficiently."""
        n = len(x)
        if n == 0:
            return
        if len(y) != n:
            raise ValueError(f"Lengths of x ({n}) and y ({len(y)}) must match")

        if n >= self._capacity:
            # Incoming data exceeds capacity; retain only the most recent capacity elements
            x_tail = x[-self._capacity :]
            y_tail = y[-self._capacity :]
            self._x[:] = x_tail
            self._y[:] = y_tail
            self._head = 0
            self._count = self._capacity
            return

        # Fits or wraps around
        end_space = self._capacity - self._head
        if n <= end_space:
            self._x[self._head : self._head + n] = x
            self._y[self._head : self._head + n] = y
            self._head = (self._head + n) % self._capacity
        else:
            first_chunk = end_space
            second_chunk = n - end_space
            self._x[self._head :] = x[:first_chunk]
            self._y[self._head :] = y[:first_chunk]
            self._x[:second_chunk] = x[first_chunk:]
            self._y[:second_chunk] = y[first_chunk:]
            self._head = second_chunk

        self._count += n

    def view(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return contiguous chronological (x, y) arrays of stored data.
        """
        valid_len = len(self)
        if valid_len == 0:
            return np.empty(0, dtype=self._dtype), np.empty(0, dtype=self._dtype)

        if self._count <= self._capacity:
            return self._x[:valid_len].copy(), self._y[:valid_len].copy()

        # Wrapped buffer: head is the oldest sample
        x_ordered = np.concatenate((self._x[self._head :], self._x[: self._head]))
        y_ordered = np.concatenate((self._y[self._head :], self._y[: self._head]))
        return x_ordered, y_ordered
