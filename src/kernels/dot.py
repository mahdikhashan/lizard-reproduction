import os

import triton
import triton.language as tl

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def block_size(n):
    return max(16, triton.next_power_of_2(n))


@triton.jit
def dot(a, b, F32: tl.constexpr):
    a = a.to(tl.bfloat16)
    b = b.to(tl.bfloat16)
    if F32:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    return tl.dot(a, b)
