import torch
import triton
import triton.language as tl
from triton.compiler import make_backend
from ..utils import get_autotune_config


allow_tf32 = not torch.version.hip
USE_ON_THE_FLY_WEIGHT_TRANSPOSE = True


def _tf32_input_precision():
    """
    tl.dot input_precision for allow_tf32. Triton's AMD backend accepts 'tf32'
    only on some targets; elsewhere use 'ieee', as when allow_tf32 is off.
    """
    if not torch.version.hip:
        return 'tf32'
    target = triton.runtime.driver.active.get_current_target()
    options = make_backend(target).parse_options({})
    return 'tf32' if 'tf32' in options.allowed_dot_input_precisions else 'ieee'


TF32_PRECISION = tl.constexpr(_tf32_input_precision())


autotune_config = get_autotune_config(
    platform={
        'cuda': [
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64}, num_stages=3, num_warps=8),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 32,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 32,  'BK': 32}, num_stages=5, num_warps=2),
            triton.Config({'B1': 32,  'B2': 64,  'BK': 32}, num_stages=5, num_warps=2),
        ],
        'hip': [
            triton.Config({'B1': 128, 'B2': 256, 'BK': 16, 'waves_per_eu': 2}, num_warps=4, num_stages=2),
            triton.Config({'B1': 256, 'B2': 256, 'BK': 16, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32, 'waves_per_eu': 3}, num_warps=4, num_stages=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32, 'waves_per_eu': 8}, num_warps=4, num_stages=2),
        ]
    },
    device={
        'A100': [
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64}, num_stages=4, num_warps=8),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 32}, num_stages=4, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64}, num_stages=4, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 32}, num_stages=4, num_warps=8),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 64}, num_stages=4, num_warps=4),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 64}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 32}, num_stages=4, num_warps=2),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32}, num_stages=4, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 32}, num_stages=4, num_warps=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64}, num_stages=4, num_warps=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 32}, num_stages=4, num_warps=2),
        ],
        'H100': [
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64}, num_stages=5, num_warps=8),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 32}, num_stages=5, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64}, num_stages=5, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 32}, num_stages=5, num_warps=8),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 64}, num_stages=5, num_warps=4),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 32}, num_stages=5, num_warps=4),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 64}, num_stages=5, num_warps=4),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 32}, num_stages=5, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 128}, num_stages=5, num_warps=8),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64}, num_stages=5, num_warps=4),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 32}, num_stages=5, num_warps=4),
        ],
        'MI300X': [
            triton.Config({'B1': 256, 'B2': 256, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 256, 'B2': 256, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 256, 'B2': 128, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=16),
            triton.Config({'B1': 128, 'B2': 256, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 32, 'waves_per_eu': 2}, num_stages=2, num_warps=8),
            triton.Config({'B1': 256, 'B2': 64,  'BK': 32, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 32, 'waves_per_eu': 2}, num_stages=2, num_warps=8),
            triton.Config({'B1': 64,  'B2': 256, 'BK': 32, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 128, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=8),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=4),
            triton.Config({'B1': 128, 'B2': 64,  'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 128, 'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=4),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64, 'waves_per_eu': 2}, num_stages=2, num_warps=2),
            triton.Config({'B1': 64,  'B2': 64,  'BK': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_stages=2, num_warps=2),
        ],
    }
)
