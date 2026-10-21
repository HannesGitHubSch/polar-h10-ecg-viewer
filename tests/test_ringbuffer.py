"""
Tests for RingBuffer circular array storage.
"""

import numpy as np
import pytest

from polar_ecg.ringbuffer import RingBuffer


def test_ringbuffer_empty():
    rb = RingBuffer(10)
    assert len(rb) == 0
    x, y = rb.view()
    assert len(x) == 0 and len(y) == 0


def test_ringbuffer_under_capacity():
    rb = RingBuffer(5)
    rb.append(1.0, 10.0)
    rb.append(2.0, 20.0)
    assert len(rb) == 2
    x, y = rb.view()
    np.testing.assert_array_equal(x, [1.0, 2.0])
    np.testing.assert_array_equal(y, [10.0, 20.0])


def test_ringbuffer_overflow_wrap():
    rb = RingBuffer(3)
    for i in range(5):
        rb.append(float(i), float(i * 10))

    assert len(rb) == 3
    x, y = rb.view()
    # Most recent 3 items should be [2, 3, 4]
    np.testing.assert_array_equal(x, [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(y, [20.0, 30.0, 40.0])


def test_ringbuffer_append_many():
    rb = RingBuffer(5)
    x_in = np.array([1.0, 2.0, 3.0])
    y_in = np.array([10.0, 20.0, 30.0])
    rb.append_many(x_in, y_in)
    assert len(rb) == 3

    # Add 4 more elements (total 7 -> wraps around)
    rb.append_many(np.array([4.0, 5.0, 6.0, 7.0]), np.array([40.0, 50.0, 60.0, 70.0]))
    assert len(rb) == 5
    x, y = rb.view()
    np.testing.assert_array_equal(x, [3.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(y, [30.0, 40.0, 50.0, 60.0, 70.0])


def test_ringbuffer_append_many_exceeds_capacity():
    rb = RingBuffer(4)
    x_in = np.arange(10, dtype=float)
    y_in = x_in * 2
    rb.append_many(x_in, y_in)

    assert len(rb) == 4
    x, y = rb.view()
    np.testing.assert_array_equal(x, [6.0, 7.0, 8.0, 9.0])
    np.testing.assert_array_equal(y, [12.0, 14.0, 16.0, 18.0])


def test_ringbuffer_clear():
    rb = RingBuffer(5)
    rb.append(1.0, 2.0)
    rb.clear()
    assert len(rb) == 0
    x, _y = rb.view()
    assert len(x) == 0


def test_ringbuffer_invalid_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)
    with pytest.raises(ValueError):
        RingBuffer(-5)
