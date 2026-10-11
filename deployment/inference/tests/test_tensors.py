"""Offline guards for raw GPU tensor descriptors, without a GPU runtime."""

from dataclasses import dataclass

import numpy as np
import pytest
from dawn_ort import DeviceTensor


@dataclass
class Owner:
    device: object
    address: int = 37
    allocated_size: int = 1040

    def retain(self) -> "Owner":
        return Owner(self.device, self.address, self.allocated_size)

    def close(self) -> None:
        self.address = 0


def test_logical_size_and_native_storage_padding_are_separate():
    device = object()
    owner = Owner(device)
    tensor = DeviceTensor(owner, (257,), np.dtype(np.float32))
    assert tensor.nbytes == 1028
    tensor.validate(device)
    owner.allocated_size = tensor.nbytes
    with pytest.raises(ValueError, match="1040 storage bytes"):
        tensor.validate(device)


@pytest.mark.parametrize("shape", [(0,), (-1,), (1.5,), (True,)])
def test_invalid_dimensions_cannot_form_a_tensor(shape):
    with pytest.raises(ValueError, match="positive integer dimensions"):
        DeviceTensor(Owner(object()), shape, np.dtype(np.float32))


def test_device_identity_and_null_handles_are_rejected():
    device = object()
    owner = Owner(device)
    tensor = DeviceTensor(owner, (1,), np.dtype(np.float32))
    with pytest.raises(ValueError, match="different runtime/device"):
        tensor.validate(object())
    owner.address = 0
    with pytest.raises(ValueError, match="null native buffer"):
        tensor.validate(device)


def test_scalar_tensor_has_one_element():
    device = object()
    tensor = DeviceTensor(Owner(device), (), np.float32)
    assert tensor.nbytes == 4
    tensor.validate(device)
