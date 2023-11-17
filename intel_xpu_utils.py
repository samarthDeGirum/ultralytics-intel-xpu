""" This file contains utility functions for Intel XPU devices. """
import torch
#import intel_extension_for_pytorch as ipex


class XpuUtils:
    def __init__(self, device_idx: int = 0):
        try:
            if torch.xpu.is_available():
                self.device_idx = device_idx
                device_str = f"xpu:{device_idx}"
                self.device = torch.device(device_str)
            else:
                raise Exception("XPU device not found")
        except Exception as e:
            print(f"Following exception occured: {e}")

    def get_num_devices(self):
        return f"Number of XPU devices found: {torch.xpu.device_count()}"

    def get_current_device(self):
        current_device_idx = torch.xpu.current_device()
        return f"Current device index: {current_device_idx}"

    def get_device_properties(self, device_idx=None):
        if device_idx is None:
            device_idx = self.device_idx
        return f"Device Properties: {torch.xpu.get_device_properties(device_idx)}"

    def get_memory_stats(self, device_idx=None):
        if device_idx is None:
            device_idx = self.device_idx
        return f"Memory stats: {itorch.xpu.memory_stats(device_idx).get_memory_stats()}"

    def get_memory_summary(self, device_idx=None):
        if device_idx is None:
            device_idx = self.device_idx
        return f"Memory summary: {torch.xpu.memory_summary(device_idx)}"

    def empty_cache(self, verbose=False):
        if verbose:
            print(torch.xpu.memory_stats(device=self.device_idx))
        if torch.xpu.is_available():
            torch.xpu.empty_cache()


## Example usage:

# xpu = XpuUtils(device_idx=0)
# print(
#    xpu.get_num_devices(),
#    xpu.get_current_device(),
#    xpu.get_device_properties(device_idx=0),
#    xpu.get_memory_stats(device_idx=0),
# )
