from typing import *
import torch
import triton


def get_gpu_name():
    return torch.cuda.get_device_name()


def get_platform_name():
    if torch.cuda.is_available():
        if getattr(torch.version, 'hip', None) is not None:
            return 'hip'
        return 'cuda'
    return 'unknown'
    

def flat_grid(num_programs):
    """
    Grid for a kernel whose program index is
    tl.program_id(0) + tl.program_id(1) * tl.num_programs(0).

    HIP rejects a launch of 2^32 or more work-items in one grid dimension (CUDA
    limits only the block count), so on HIP the programs are spread over a second
    axis whenever a work-group of up to 1024 work-items could reach that limit.
    The last row may overrun num_programs; the kernel must mask those programs.
    """
    if get_platform_name() != 'hip':
        return (num_programs,)
    rows = max(1, triton.cdiv(num_programs, (2**32 - 1) // 1024))
    return (triton.cdiv(num_programs, rows), rows)


def get_num_sm():
    return torch.cuda.get_device_properties("cuda").multi_processor_count
    

def get_autotune_config(
    default: List[triton.Config] = None,
    platform: Dict[str, List[triton.Config]] = None,
    device: Dict[str, List[triton.Config]] = None,
) -> List[triton.Config]:
    """
    Get the autotune configuration for the current platform and device.
    """
    if device is not None:
        gpu_name = get_gpu_name()
        for key, value in device.items():
            if key.lower() in gpu_name.lower():
                return value
    
    if platform is not None:
        platform_name = get_platform_name()
        for key, value in platform.items():
            if key.lower() in platform_name.lower():
                return value
    
    if default is None:
        raise ValueError("No autotune configuration found for the current platform and device.")
    return default
