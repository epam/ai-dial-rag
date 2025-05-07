from enum import StrEnum

import torch


class DeviceType(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


def autodetect_device() -> DeviceType:
    """Detects the device type based on the availability of CUDA."""
    if torch.cuda.is_available():
        return DeviceType.CUDA
    return DeviceType.CPU


def detect_device(device_str: str) -> DeviceType:
    """Parses the device type from a string. Detects the device if the string is 'auto'."""
    if device_str == DeviceType.AUTO:
        device = autodetect_device()
    elif device_str in DeviceType:
        device = DeviceType(device_str)
    else:
        raise ValueError(f"Unknown device type: {device_str}")

    return device
